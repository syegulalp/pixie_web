from pixie_web import route, run, Response, RouteType, proc_env

# Local synchronous
@route("/", RouteType.sync)
def index(env):
    return Response(f"Hello world from process type {proc_env.proc_type}")


# Local async
@route("/async", RouteType.asnc)
async def index_async(env):
    return Response(f"Hello world (async) from process type {proc_env.proc_type}")


# Process-pooled (the default)
@route("/cpu", RouteType.pool)
def cpu_bound(env):
    # from time import sleep

    # sleep(3)
    return Response(f"Hello world (CPU-bound) from process type {proc_env.proc_type}")


if __name__ == "__main__":
    run()
