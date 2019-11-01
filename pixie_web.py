from typing import Optional, Union, Callable

DEFAULT_TIMEOUT = 15.0
FILE_CACHE_SIZE = 256
TEMPLATE_CACHE_SIZE = 256

import pickle

pickle.DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL

import asyncio
import re

from sys import stderr
from os import getcwd as os_getcwd, stat as os_stat
from os.path import join as path_join
from functools import lru_cache
from email.utils import formatdate
from urllib.parse import unquote
from mimetypes import guess_type
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue, Manager, Event
from queue import Empty as EmptyQueue
from enum import Enum
from asyncio.exceptions import CancelledError

pool = None
mgr = None


class ProcessType(Enum):
    """
    Classifications for the different ways routes can be processed.
    """

    main = 0
    main_async = 1
    stream = 2
    process_pool = 3


def _e(msg: str):
    """
    Shortcut for output to stderr.
    """
    print(msg, file=stderr)


static_routes = {}
dynamic_routes = []

http_codes = {
    200: "OK",
    404: "Not Found",
    451: "Unavailable For Legal Reasons",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


class Env:
    """
    Environment object created from a HTTP request.
    """

    def __init__(
        self,
        headers: Optional[bytes] = None,
        proc_type: ProcessType = ProcessType.main,
    ):
        self.raw_headers = headers
        self.headers = self._form = None
        self.proc_type = proc_type

    def as_dict(self):
        """
        Provide headers as a dictionary.
        Use cached copy if available.
        """
        if self.headers:
            return self.headers

        if b"\r" in self.raw_headers:
            data = self.raw_headers.decode("utf-8").split("\r\n\r\n")
            hdr = data[0].split("\r\n")
        else:
            data = self.raw_headers.decode("utf-8").split("\n\n")
            hdr = data[0].split("\n")

        header = hdr.pop(0).strip().split(" ")

        self.headers = {
            "_VERB": header[0],
            "_PATH": header[1],
            "_PROTOCOL": header[2],
        }

        for _ in hdr:
            _ = _.split(":", 1)
            try:
                self.headers[_[0]] = _[1].strip()
            except IndexError:
                pass

        self.body = data[1]

        return self.headers

    def form(self):
        """
        Provide form data as a dictionary, if available.
        Returns None if the request has no form data.
        Use a cached copy if we can.
        """
        if self._form:
            return self._form

        self.as_dict()

        if self.headers.get("Content-Type") != "application/x-www-form-urlencoded":
            return None

        if "\r" in self.body:
            form = self.body.split("\r\n")
        else:
            form = self.body.split("\n")

        self._form = {}

        for _ in form:
            __ = _.split("=")
            self._form[__[0]] = unquote(__[1])

        return self._form


env = Env()


@lru_cache(TEMPLATE_CACHE_SIZE)
def template(path: str, root: str = os_getcwd()) -> str:
    """
    Load a cachable copy of an on-disk file for use as a template.
    """
    with open(path_join(root, path), "r") as file:
        return file.read()


def template_uncached(path: str) -> str:
    """
    Load template file, bypassing in-memory cache.
    """
    return template.__wrapped__(path)


def static_file(path: str, root: str = os_getcwd(), max_age: int = 38400) -> bytes:
    """
    Load static file from `path`, using `root` as the directory to start at.
    """
    file_type, encoding = guess_type(path)
    full_path = path_join(root, path)
    try:
        with open(full_path, "rb") as file:
            stats = os_stat(full_path)
            last_mod = formatdate(stats.st_mtime, usegmt=True)
            return response(
                file.read(),
                content_type=file_type,
                headers={
                    "Cache-Control": f"private, max-age={max_age}",
                    "Last-Modified": last_mod,
                },
            )
    except FileNotFoundError:
        return error_404(full_path)


@lru_cache(FILE_CACHE_SIZE)
def cached_file(path: str, root: str = os_getcwd(), max_age: int = 38400) -> bytes:
    """
    Load a static file, but use the in-memory cache to avoid roundtrips to disk.
    """
    return static_file(path, root, max_age)


def response(
    body: Union[str, bytes],
    code: int = 200,
    content_type: str = "text/html",
    headers: Optional[dict] = None,
) -> bytes:
    """
    Generate a response object (a byte stream) from either a string or a bytes object. Use `content_type` to set the Content-Type: header, `code` to set the HTTP response code, and pass a dict to `headers` to set other headers as needed.
    """

    if body is None:
        body = b""
    else:
        if type(body) is str:
            body = body.encode("utf-8")
        length = len(body)
        if not headers:
            headers = {}
        headers["Content-Length"] = length

    if headers is not None:
        headers = "\n" + "\n".join([f"{k}: {v}" for k, v in headers.items()])
    else:
        headers = ""

    return (
        bytes(
            f"HTTP/1.1 {code} {http_codes[code]}\nContent-Type: {content_type}{headers}\n\n",
            "utf-8",
        )
        + body
    )


def header(
    code: int = 200, content_type: str = "text/html", headers: Optional[dict] = None
):
    return response(None, code, content_type, headers)


def error_404(path: str) -> bytes:
    """
    Built-in 404: Not Found error handler.
    """
    return response(f"<h1>Not found: {path}</h1>", 404)


def error_500(path: str, error: Exception) -> bytes:
    """
    Built-in 500: Server Error handler.
    """
    return response(f"<h1>Server error in {path}</h1><p>{str(error)}</p>", 500)


def error_503(path: str) -> bytes:
    """
    Built-in 503: Server Timeout handler.
    """
    return response(
        f"<h1>Server timed out after {DEFAULT_TIMEOUT} seconds in {path}</h1>", 503
    )


path_re_str = "<([^>]*)>"
path_re = re.compile(path_re_str)


def route(path: str, proc_type: ProcessType = ProcessType.process_pool):
    """
    Route decorator, used to assign a route to a function handler by wrapping the function. Accepts a `path` and an optional `proc_type` as arguments.
    """
    parameters = []
    route_regex = None

    path_match = re.finditer(path_re, path)

    for n in path_match:
        parameters.append(n.group(0)[1:-1])

    if parameters:
        route_regex = re.compile(re.sub(path_re_str, "(.*?)", path))

    def decorator(callback):
        if route_regex:
            add_dynamic_route(route_regex, callback, proc_type, parameters)
        else:
            add_route(path, callback, proc_type)
        return callback

    return decorator


def add_route(
    path: str, callback: Callable, proc_type: ProcessType = ProcessType.process_pool
):
    """
    Assign a static route to a function handler.
    """
    static_routes[path] = (callback, proc_type)


def add_dynamic_route(
    path: str,
    callback: Callable,
    proc_type: ProcessType = ProcessType.process_pool,
    parameters: list = None,
):
    """
    Assign a dynamic route (with wildcards) to a function handler.
    """
    dynamic_routes.append((path, callback, proc_type, parameters))


def run_route_pool(raw_env: bytes, func: Callable, *a, **ka):
    """
    Execute a function synchronously in the local environment. A copy of the HTTP request data is passed automatically to the handler as its first argument.
    """
    local_env = Env(raw_env, proc_type=env.proc_type)
    return func(local_env, *a, **ka)


def run_route_pool_stream(
    remote_queue: Queue, signal: Event, raw_env: bytes, func: Callable, *a, **ka
):
    """
    Execute a function synchronously in the process pool, and return results from it incrementally.
    """
    local_env = Env(raw_env, proc_type=env.proc_type)
    for _ in func(local_env, *a, **ka):
        if signal.is_set():
            raise Exception
        remote_queue.put(_)
    remote_queue.put(b"")


async def connection_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
):
    """
    Reeads the data from the network connection, and attempts to find an appropriate route for it.
    """

    readline = reader.readline
    get_loop = asyncio.get_event_loop
    write = writer.write
    drain = writer.drain
    at_eof = reader.at_eof
    wait_for = asyncio.wait_for
    close = writer.close
    AsyncTimeout = asyncio.exceptions.TimeoutError
    run_in_executor = get_loop().run_in_executor

    while True:

        action = raw_data = signal = None
        content_length = 0

        while True:
            _ = await readline()

            if at_eof():
                close()
                return

            if raw_data is None:
                raw_data = _
                action = _.decode("utf-8").split(" ")
                continue
            else:
                raw_data += _

            if _ in (b"\r\n", b"\n"):
                break

            if b"Content-Length:" in _:
                content_length = int(_.decode("utf-8").split(":")[1])

        raw_data += await reader.read(content_length)

        path = action[1].split("?", 1)[0]

        try:
            handler, pool_type = static_routes[path]
        except KeyError:
            for route in dynamic_routes:
                route_match = route[0].fullmatch(path)
                if route_match:
                    handler, pool_type = route[1:3]
                    parameters = route_match.groups()
            if not route_match:
                write(error_404(path))
                await drain()
                continue
        else:
            parameters = []

        try:

            # Run with no pooling or async, in default process
            # Single-threaded, potentially blocking

            if pool_type is ProcessType.main:
                result = handler(Env(raw_data, proc_type=pool_type), *parameters)

            # Run as async, in default process
            # Single-threaded, nonblocking

            elif pool_type is ProcessType.main_async:
                result = await handler(Env(raw_data, proc_type=pool_type), *parameters)

            # Run non-async code in process pool
            # Multi-processing, not blocking

            elif pool_type is ProcessType.process_pool:
                result = await wait_for(
                    run_in_executor(
                        pool, run_route_pool, raw_data, handler, *parameters
                    ),
                    DEFAULT_TIMEOUT,
                )

            # Run incremental stream, potentially blocking, in process pool

            elif pool_type is ProcessType.stream:

                job_queue = mgr.Queue()
                signal = mgr.Event()

                job = pool.submit(
                    run_route_pool_stream,
                    job_queue,
                    signal,
                    raw_data,
                    handler,
                    *parameters,
                )

                writer.transport.set_write_buffer_limits(0)

                # We can't send an async queue object to the subprocess,
                # so we use a manager queue and poll it every .1 sec

                while True:

                    while True:
                        try:
                            _ = job_queue.get_nowait()
                        except EmptyQueue:
                            await asyncio.sleep(0.1)
                            continue
                        else:
                            break

                    write(_)
                    await drain()

                    if _ == b"":
                        break

                writer.close()
                return

        except AsyncTimeout:
            result = error_503(path)

        except Exception as err:
            result = error_500(path, err)

        try:
            if result is None:
                write(response(b""))
                await drain()

            elif isinstance(result, bytes):
                write(result)
                await drain()

            else:
                for _ in result:
                    write(_)
                    await drain()
                writer.close()
                return

        except ConnectionAbortedError:
            if signal:
                signal.set()
            writer.close()
            return


# Startup operations


def pool_start():
    """
    Launched at the start of each pooled process. This modifies the environment data in the process to let any routes running in the process know that it's in a pool, not in the main process.
    """
    env.proc_type = ProcessType.process_pool
    # global job_queue
    # job_queue = queue_to_use


def dummy():
    pass


def use_process_pool(workers: Optional[int] = None):
    """
    Set up the process pool and ensure it's running correctly.
    """
    # global job_queue
    # job_queue = Queue()

    global mgr
    mgr = Manager()

    global pool
    pool = ProcessPoolExecutor(max_workers=workers, initializer=pool_start)

    from concurrent.futures.process import BrokenProcessPool

    try:
        pool.submit(dummy).result()
    except (OSError, RuntimeError, BrokenProcessPool):
        _e(
            "'run()' function must be invoked from within 'if __name__ == \"__main__\"' block to invoke multiprocessing. Defaulting to single-process pool."
        )
        pool = None
    else:
        _e(f"Using {pool._max_workers} processes")


srv = None


def close_server():
    global srv
    if srv is None:
        raise Exception(
            "No server to close on this instance. Use `ProcessType.main_async` to route the close operation to the main server."
        )
    srv.close()
    srv = None


async def start_server(host: str, port: int):
    """
    Launch the asyncio server with the master connection handler.
    """
    global srv
    srv = await asyncio.start_server(connection_handler, host, port)
    async with srv:
        _e(f"Listening on {host}:{port}")
        await srv.serve_forever()


def run(
    host: str = "localhost", port: int = 8000, workers: Optional[bool] = True,
):
    """
    Run pixie_web on the stated hostname and port.
    """
    _e("Pixie-web 0.1")

    if workers is not None:
        if workers is True:
            use_process_pool()
        else:
            use_process_pool(int(workers))

    try:
        asyncio.run(start_server(host, port))
    except KeyboardInterrupt:
        _e("Closing server with ctrl-C")
