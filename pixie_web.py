from typing import Optional, Union, Callable, Type, Iterable

DEFAULT_TIMEOUT = 15.0
FILE_CACHE_SIZE = 256
TEMPLATE_CACHE_SIZE = 256

import re
import pickle

pickle.DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL

import asyncio
import html

try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from sys import stderr
from os import getcwd as os_getcwd, stat as os_stat
from os.path import join as path_join
from functools import lru_cache
from email.utils import formatdate
from urllib.parse import unquote_plus
from http import cookies as http_cookies
from mimetypes import guess_type
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue, Manager, Event
from queue import Empty as EmptyQueue
from enum import Enum

pool = None
mgr = None


class ParentProcessConnectionAborted(ConnectionAbortedError):
    pass


class ProcessType(Enum):
    """
    Classifications for the different process types.
    """

    main = 0
    pool = 1


class RouteType(Enum):
    """
    Classifications for the different ways routes can be processed.
    `sync`: Synchronous route.
    `sync_thread`: Sync route using thread pool.
    `asnc`: Async route.
    `pool`: Multiprocessing-pooled route.
    `stream`: Multiprocessing-pooled route that yields results incrementally (and therefore may block quite a bit).
    """

    sync = 0
    sync_thread = 1
    asnc = 2
    pool = 3
    stream = 4


def _e(msg: str):
    """
    Shortcut for output to stderr.
    """
    print(msg, file=stderr)


http_codes = {
    200: "OK",
    404: "Not Found",
    451: "Unavailable For Legal Reasons",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


class ProcEnv:
    """
    Describes the process type for the running process. If you import proc_env you can inspect proc_type and pool to see what type of process your function is running in, and whether or not your have access to the process pool. (You only have access to the process pool from the main thread.)
    """

    def __init__(
        self,
        proc_type: Enum = ProcessType.main,
        pool: Optional[ProcessPoolExecutor] = None,
    ):
        self.proc_type = proc_type
        self.pool = pool


class Request:
    """
    Object created from a HTTP request.
    """

    def __init__(self, headers: bytes, init=False):
        self.raw_data = headers
        self._headers = self._form = self._body = None
        if init:
            self.headers()

    @property
    def headers(self):
        """
        Provide headers as a dictionary.
        Use cached copy if available.
        """
        if self._headers:
            return self._headers

        self._headers = {}

        if b"\r" in self.raw_data:
            data = self.raw_data.decode("utf-8").split("\r\n\r\n")
            request = data[0].split("\r\n")
        else:
            data = self.raw_data.decode("utf-8").split("\n\n")
            request = data[0].split("\n")

        self.request = request.pop(0).strip().split(" ")

        for _ in request:
            _ = _.split(":", 1)
            try:
                self._headers[_[0]] = _[1].strip()
            except IndexError:
                pass

        self._body = data[1]

        return self._headers

    @property
    def req(self):
        try:
            return self.request
        except:
            self.headers
            return self.request

    @property
    def verb(self):
        try:
            return self.request[0]
        except:
            self.headers
            return self.request[0]

    @property
    def path(self):
        try:
            return self.request[1]
        except:
            self.headers
            return self.request[1]

    @property
    def protocol(self):
        try:
            return self.request[2]
        except:
            self.headers
            return self.request[2]

    @property
    def body(self):
        """
        Provide body as bytes.
        Use cached copy if available.
        """

        if self._body:
            return self._body

        self.headers

        return self._body

    @property
    def form(self):
        """
        Provide form data as a dictionary, if available.
        Returns None if the request has no form data.
        Use a cached copy if we can.
        """
        if self._form:
            return self._form

        if self.headers.get("Content-Type") != "application/x-www-form-urlencoded":
            return None

        if "\r" in self._body:
            form = self._body.split("\r\n")
        else:
            form = self._body.split("\n")

        self._form = {}

        for _ in form:
            __ = _.split("=")
            self._form[__[0]] = unquote_plus(__[1])

        return self._form


class Response:
    """
    Generate a complex response object (a class instance) from a string object. Use `content_type` to set the Content-Type: header, `code` to set the HTTP response code, and pass a dict to `headers` to set other headers as needed.

    Use this when you want to perform complex manipulations on a response, like mutating its properties across multiple functions, before returning it to the client.
    """

    def __init__(
        self,
        body: str = "",
        code: int = 200,
        content_type: str = "text/html",
        headers: Optional[dict] = None,
        cookies: Optional[dict] = None,
    ):
        self.body = body
        self.headers = headers
        self.code = code
        self.content_type = content_type

        if cookies is not None:
            self.cookies: Optional[
                http_cookies.SimpleCookie
            ] = http_cookies.SimpleCookie()
            for k, v in cookies.items():
                self.cookies[k] = v
        else:
            self.cookies = None

    def as_bytes(self) -> bytes:
        return simple_response(
            self.body, self.code, self.content_type, self.headers, self.cookies
        )


class Unsafe:
    def __init__(self, data: str):
        self.data = data

    @property
    def esc(self):
        return html.escape(self.data)


class Template:
    def __init__(self, template: str = None, filename=None, path=None):
        if filename:
            with open(path_join(path, filename), "r") as f:
                template = f.read()
        self.template = template

    def render(self, *a, **ka):
        if a:
            new_a = []
            for _ in a:
                if isinstance(_, Unsafe):
                    new_a.append(_.data)
                else:
                    new_a.append(html.escape(str(_)))
            return self.template.format(*new_a)

        if ka:
            for k, v in ka.items():
                if isinstance(v, Unsafe):
                    ka[k] = v.data
                else:
                    ka[k] = html.escape(str(v))
            return self.template.format(**ka)


def template(string, *a, **ka):
    return Template(string).render(*a, **ka)


def static_file(
    filename: str, path: str = os_getcwd(), max_age: int = 38400
) -> Union[bytes, Response]:
    """
    Load static file from `path`, using `root` as the directory to start at.
    """
    file_type, encoding = guess_type(filename)
    full_path = path_join(path, filename)
    try:
        with open(full_path, "rb") as file:
            stats = os_stat(full_path)
            last_mod = formatdate(stats.st_mtime, usegmt=True)
            return simple_response(
                file.read(),
                content_type=file_type,
                headers={
                    "Cache-Control": f"private, max-age={max_age}",
                    "Last-Modified": last_mod,
                },
            )
    except FileNotFoundError as e:
        raise e


@lru_cache(FILE_CACHE_SIZE)
def cached_file(
    path: str, root: str = os_getcwd(), max_age: int = 38400
) -> Union[bytes, Response]:
    """
    Load a static file, but use the in-memory cache to avoid roundtrips to disk.
    """
    return static_file(path, root, max_age)


def simple_response(
    body: Union[str, bytes, None],
    code: int = 200,
    content_type: Optional[str] = "text/html",
    headers: Optional[dict] = None,
    cookies: Optional[http_cookies.SimpleCookie] = None,
) -> bytes:
    """
    Generate a simple response object (a byte stream) from either a string or a bytes object. Use `content_type` to set the Content-Type: header, `code` to set the HTTP response code, and pass a dict to `headers` to set other headers as needed.

    Use this when you want to simply return a byte sequence as your response, without needing to manipulate the results too much. You can also use this to `yield` headers, and then pieces of a body (as simple `bytes` objects), when you want to return results incrementally.
    """

    if body is None:
        body = b""
    else:
        if type(body) is str:
            body = body.encode("utf-8")  # type: ignore
        length = len(body)
        if not headers:
            headers = {}
        headers["Content-Length"] = length

    if headers is not None:
        header_str = "\r\n" + "\r\n".join([f"{k}: {v}" for k, v in headers.items()])
    else:
        header_str = ""

    if cookies is not None:
        cookie_str = "\r\n" + cookies.output()
    else:
        cookie_str = ""

    return (
        bytes(  # type: ignore
            f"HTTP/1.1 {code} {http_codes[code]}\r\nContent-Type: {content_type}{header_str}{cookie_str}\r\n\r\n",
            "utf-8",
        )
        + body
    )


def header(
    code: int = 200, content_type: str = "text/html", headers: Optional[dict] = None
):
    return simple_response(None, code, content_type, headers)


path_re_str = "<([^>]*)>"
path_re = re.compile(path_re_str)


def dummy():
    pass


class Server:
    def __init__(self):
        self.static_routes: dict = {}
        self.dynamic_routes = []
        self.pool: Optional[ProcessPoolExecutor] = None
        self.proc_env = ProcEnv()

    template_404 = Template("<h1>Path or file not found: {}</h1>")
    template_500 = Template("<h1>Server error in {}</h1><p>{}")
    template_503 = Template("<h1>Server timed out after {} seconds in {}</h1>")

    def error_404(self, request: Request) -> bytes:
        """
        Built-in 404: Not Found error handler.
        """
        return simple_response(self.template_404.render(request.path), code=404)

    def error_500(self, request: Request, error: Exception) -> bytes:
        """
        Built-in 500: Server Error handler.
        """
        return simple_response(
            self.template_500.render(request.path, str(error)), code=500,
        )

    def error_503(self, request: Request) -> bytes:
        """
        Built-in 503: Server Timeout handler.
        """
        return simple_response(
            self.template_503.render(DEFAULT_TIMEOUT, request.path), code=503,
        )

    def route(
        self,
        path: str,
        route_type: RouteType = RouteType.pool,
        action: Union[Iterable, str] = "GET",
        before=None,
        after=None,
    ):
        """
        Route decorator, used to assign a route to a function handler by wrapping the function. Accepts a `path`, an optional `route_type`, and an optional list of HTTP verbs (or a single verb string, default "GET") as arguments.
        """
        parameters = []
        route_regex = None

        path_match = re.finditer(path_re, path)

        for n in path_match:
            parameters.append(n.group(0)[1:-1])

        if parameters:
            route_regex = re.compile(re.sub(path_re_str, "(.*?)", path))

        if isinstance(action, str):
            action = [action]

        def decorator(callback):

            if route_regex:
                for _ in action:
                    self.add_dynamic_route(
                        route_regex, _, callback, route_type, parameters
                    )
            else:
                for _ in action:
                    self.add_route(path, _, callback, route_type)
            return callback

        return decorator

    def add_route(
        self,
        path: str,
        action: str,
        callback: Callable,
        route_type: RouteType = RouteType.pool,
    ):
        """
        Assign a static route to a function handler.
        """
        route = (callback, route_type)
        if not self.static_routes.get(path):
            self.static_routes[path] = {action: route}
        else:
            self.static_routes[path][action] = route

    def add_dynamic_route(
        self,
        regex_pattern,
        action: str,
        callback: Callable,
        route_type: RouteType = RouteType.pool,
        parameters: list = None,
    ):
        """
        Assign a dynamic route (with wildcards) to a function handler.
        """
        self.dynamic_routes.append(
            (regex_pattern, action, callback, route_type, parameters)
        )

    @classmethod
    def run_route_pool(cls, raw_env: bytes, func: Callable, *a, **ka):
        """
        Execute a function synchronously in the local environment. A copy of the HTTP request data is passed automatically to the handler as its first argument.
        """
        local_env = Request(raw_env)
        result = func(local_env, *a, **ka)
        if isinstance(result, Response):
            return result.as_bytes()
        return result

    @classmethod
    def run_route_pool_stream(
        cls, remote_queue: Queue, signal, raw_env: bytes, func: Callable, *a, **ka
    ):
        """
        Execute a function synchronously in the process pool, and return results from it incrementally.
        """
        local_env = Request(raw_env)
        for _ in func(local_env, *a, **ka):
            if signal.is_set():
                raise ParentProcessConnectionAborted
            remote_queue.put(_)
        remote_queue.put(None)

    async def start_server(self, host: str, port: int):
        """
        Launch the asyncio server with the master connection handler.
        """
        self.srv = await asyncio.start_server(self.connection_handler, host, port)
        async with self.srv:  # type: ignore
            _e(f"Listening on {host}:{port}")
            await self.srv.serve_forever()

    def run(
        self,
        host: str = "localhost",
        port: int = 8000,
        workers: Union[bool, int, None] = True,
    ):
        """
        Run pixie_web on the stated hostname and port.
        """
        _e("Pixie-web 0.1")

        if workers is not None:
            if workers is True:
                self.use_process_pool()
            elif workers is False:
                pass
            else:
                self.use_process_pool(int(workers))

        try:
            asyncio.run(self.start_server(host, port))
        except KeyboardInterrupt:
            _e("Closing server with ctrl-C")
        except asyncio.CancelledError:
            _e("Closing due to internal loop shutdown")

    @classmethod
    def pool_start(cls):
        """
        Launched at the start of each pooled process. This modifies the environment data in the process to let any routes running in the process know that it's in a pool, not in the main process.
        """

        proc_env.proc_type = ProcessType.pool

    def use_process_pool(self, workers: Optional[int] = None):
        """
        Set up the process pool and ensure it's running correctly.
        """
        self.mgr = Manager()

        self.pool = ProcessPoolExecutor(
            max_workers=workers, initializer=Server.pool_start
        )

        from concurrent.futures.process import BrokenProcessPool

        try:
            self.pool.submit(dummy).result()
        except (OSError, RuntimeError, BrokenProcessPool):
            _e(
                "'run()' function must be invoked from within 'if __name__ == \"__main__\"' block to invoke multiprocessing. Defaulting to single-process pool."
            )
            self.pool = None
        else:
            _e(f"Using {self.pool._max_workers} processes")  # type: ignore

        self.proc_env.pool = self.pool

    def close_server(self):
        if self.srv is None:
            raise Exception(
                "No server to close on this instance. Use `ProcessType.main_async` to route the close operation to the main server."
            )
        self.srv.close()
        self.srv = None

    async def connection_handler(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
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
        AsyncTimeout = asyncio.TimeoutError
        run_in_executor = get_loop().run_in_executor

        while True:

            action = raw_data = signal = content_length = None

            while True:
                _ = await readline()

                if at_eof():
                    close()
                    return

                if raw_data is None:
                    raw_data = bytearray(_)
                    action = _.decode("utf-8").split(" ")
                    continue
                else:
                    raw_data.extend(_)

                if _ in (b"\r\n", b"\n"):
                    break

                if _.startswith(b"Content-Length:"):
                    content_length = int(_.decode("utf-8").split(":")[1])

            if content_length:
                raw_data.extend(await reader.read(content_length))

            path = action[1].split("?", 1)[0]
            verb = action[0]

            try:
                handler, route_type = self.static_routes[path][verb]
            except KeyError:
                for route in self.dynamic_routes:
                    if verb != route[1]:
                        continue
                    route_match = route[0].fullmatch(path)
                    if route_match:
                        handler, route_type = route[2:4]
                        parameters = route_match.groups()
                if (not self.dynamic_routes) or (not route_match):
                    write(self.error_404(Request(raw_data)))
                    await drain()
                    continue
            else:
                parameters = []

            try:

                # Run with no pooling or async, in default process.
                # Single-threaded, potentially blocking.

                if route_type is RouteType.sync:
                    result = handler(Request(raw_data), *parameters)

                # Run a sync function in an async thread (cooperative multitasking)

                elif route_type is RouteType.sync_thread:
                    result = await run_in_executor(
                        None, handler, Request(raw_data), *parameters
                    )

                # Run async function in default process.
                # Single-threaded, nonblocking.

                elif route_type is RouteType.asnc:
                    result = await handler(Request(raw_data), *parameters)

                # Run non-async code in process pool.
                # Multi-processing, not blocking.

                # Note that we pass `Server.run_route_pool`, not `self.run_route_pool`, because otherwise we can't correctly pickle the object. So we just use the class method that exists in the pool instance, since it doesn't need `self` anyway. If we DID need `self` over there, we could always get the server instance from the module-local server obj.

                elif route_type is RouteType.pool:
                    result = await wait_for(
                        run_in_executor(
                            self.pool,
                            Server.run_route_pool,
                            raw_data,
                            handler,
                            *parameters,
                        ),
                        DEFAULT_TIMEOUT,
                    )

                # Run incremental stream, potentially blocking, in process pool

                elif route_type is RouteType.stream:

                    job_queue = self.mgr.Queue()
                    signal = self.mgr.Event()

                    job = self.pool.submit(
                        Server.run_route_pool_stream,
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

                        if _ is None:
                            break

                        write(_)
                        await drain()

                    writer.close()
                    return

            except FileNotFoundError:
                result = self.error_404(Request(raw_data))
            except AsyncTimeout:
                result = self.error_503(Request(raw_data))
            except Exception as err:
                result = self.error_500(Request(raw_data), err)

            try:
                if isinstance(result, Response):
                    write(result.as_bytes())
                elif isinstance(result, bytes):
                    write(result)
                elif result is None:
                    write(simple_response(b""))
                else:
                    for _ in result:
                        write(_)
                        await drain()
                    writer.close()
                    return

                await drain()

            except ConnectionAbortedError:
                if signal:
                    signal.set()
                writer.close()
                return


server = Server()
route = server.route
run = server.run
proc_env = server.proc_env
