from pixie_web import (
    route,
    run,
    Response,
    header,
    proc_env,
    static_file,
    cached_file,
    Template,
    RouteType,
    Literal,
)

from asyncio import sleep as asyncio_sleep
from time import sleep

body_template = Template(filename="template.html", path="demo")


def body(request, text):
    return body_template.render(
        header=text,
        request_id=id(request),
        env_id=id(proc_env),
        proc_env=proc_env.proc_type,
        env_var=request.headers.items(),
        form_data=request.form,
    )


# Site root.
# Runs single-threaded in the local process.


@route("/", action=("GET", "POST"))
def index(env):
    return Response(body(env, "Pixie is running!"))


# Serve a single static file from the /demo directory.


@route("/static_file", RouteType.sync, action=("GET", "POST"))
def _static_file(env):
    return static_file("index.html", "demo")


# Serve a memory-cached static file from the /demo directory.


@route("/cached_file", RouteType.sync, action=("GET", "POST"))
def cached_static_file(env):
    return cached_file("index.html", "demo")


# Serve a memory-cached file by name using a wildcard route.


@route("/file/<filename>", RouteType.sync, action=("GET", "POST"))
def cached_routed_file(env, filename):
    return cached_file(filename, "demo")


# Generate an error (in a subprocess).


@route("/error", action=("GET", "POST"))
def crash(env):
    raise Exception("An exception occurred in a pool process!")


# Generate an error (in the main process).


@route("/error-local", RouteType.sync, action=("GET", "POST"))
def crash_local(env):
    raise Exception("An exception occurred in the main process!")


# Run a route using async.


@route("/async", RouteType.asnc, action=("GET", "POST"))
async def async_pool(env):
    return Response(body(env, "Async"))


# Run a blocking sync function in an async thread (cooperative multitasking).


@route("/thread", RouteType.sync_thread, action=("GET", "POST"))
def async_thread_pool(env):
    sleep(3)
    return Response(body(env, "Sync in async thread (slept for 3 seconds)"))


# Run a local, single-threaded, sync process (same as /).


@route("/local", RouteType.sync, action=("GET", "POST"))
def local_proc(env):
    return Response(body(env, "Native process"))


# Run a process in the process pool.


@route("/proc", RouteType.pool, action=("GET", "POST"))
def proc_pool(env):
    return Response(body(env, "Process pool (explicitly labeled)"))


# Run a route that times out but doesn't block (since it's running in the process pool).


@route("/sleep-timeout", action=("GET", "POST"))
def sleep_timeout(env):
    sleep(30)
    return Response(body(env, "Slept for 30 seconds"))


# Run a route that is CPU-intensive in the process pool.


@route("/sleep", action=("GET", "POST"))
def sleep_short(env):
    sleep(3)
    return Response(body(env, "Slept for 3 seconds"))


# Sleep for 3 seconds in an async route.
# You must use `asyncio_sleep` for this to work in async.


@route("/sleep-async", RouteType.asnc, action=("GET", "POST"))
async def sleep_async(env):
    await asyncio_sleep(3)
    return Response(body(env, "Slept for 3 seconds (async)"))


# Yield blocking results incrementally (runs in subprocess).
# Not recommended but we can do it.


@route("/stream", RouteType.stream, action=("GET", "POST"))
def stream(env):
    # Send header-only first
    yield header()

    # Send body incrementally
    for _ in range(10):
        yield bytes(f"<p>This is part {_} of 10.</p>", "utf-8")
        sleep(1)


# The server cancellation must be in the main process. Note that you can use `yield` to return a message before the server shuts down.


@route("/shutdown", RouteType.sync)
def end_server(env):
    from pixie_web import server

    yield header()
    yield b"Server closed"
    server.close_server()


if __name__ == "__main__":
    try:
        run(port=80)
    except Exception as e:
        print(e)
