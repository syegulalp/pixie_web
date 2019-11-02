from pixie_web import route, run, response, RouteType, pool_env

# Local synchronous


@route("/", RouteType.sync)
def index(env):
    return response("Hello world")


# Local async


@route("/async", RouteType.asnc)
async def index_async(env):
    return response("Hello world async")


# Process-pooled (the default)


@route("/cpu", RouteType.pool)
def cpu_bound(env):
    from time import sleep

    sleep(3)
    return response("Hello world (CPU-bound operation)")


if __name__ == "__main__":
    run()
