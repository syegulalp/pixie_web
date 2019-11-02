from pixie_web import (
    route,
    run,
    response,
    header,
    pool_env,
    static_file,
    cached_file,
    template,
    RouteType
)

from time import sleep

# Load template from file on disk.
# Note that if we call `template()` in each route function,
# the results will be cached in-memory automatically,
# but we can load once in the body of the function too.

body_template = template("template.html", "demo")


def body(env, text):
    return body_template.format(
        text,
        id(env),
        id(pool_env),
        pool_env.proc_type,
        "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in env.headers.items()]),
        env.form,
    )


# Site root
# Runs single-threaded in the local process


@route("/", RouteType.sync)
def index(env):
    return response(body(env, "Pixie is running!"))


# Serve a single static file from the /demo directory


@route("/static_file", RouteType.sync)
def _static_file(env):
    return static_file("index.html", "demo")


# Serve a memory-cached static file from the /demo directory


@route("/cached_file", RouteType.sync)
def cached_static_file(env):
    return cached_file("index.html", "demo")


# Serve a memory-cached file by name using a wildcard route


@route("/file/<filename>", RouteType.sync)
def cached_routed_file(env, filename):
    return cached_file(filename, "demo")


# Generate an error (in a subprocess)


@route("/error")
def crash(env):
    raise Exception(body(env, "An exception occurred in a pool process!"))


# Generate an error (in the main process)


@route("/error-local", RouteType.sync)
def crash_local(env):
    raise Exception(body(env, "An exception occurred in the main process!"))


# Run a route using async


@route("/async", RouteType.asnc)
async def async_pool(env):
    return response(body(env, "Async"))


# Run a local, single-threaded, sync process (same as /)


@route("/local", RouteType.sync)
def local_proc(env):
    return response(body(env, "Native process"))


# Run a process in the process pool


@route("/proc", RouteType.pool)
def proc_pool(env):
    return response(body(env, "Process pool (explicitly labeled)"))


# Run a route that times out but doesn't block (since it's running in the process pool).


@route("/sleep-timeout")
def sleep_timeout(env):
    sleep(30)
    return response(body(env, "Slept for 30 seconds"))


# Run a route that is CPU-intensive in the process pool


@route("/sleep")
def sleep_short(env):
    sleep(3)
    return response(body(env, "Slept for 3 seconds"))


# Yield blocking results incrementally (runs in subprocess)
# Not recommended but we can do it.


@route("/stream", RouteType.stream)
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
    from pixie_web import close_server

    yield response("Server closed")
    close_server()


if __name__ == "__main__":
    run(port=80)
