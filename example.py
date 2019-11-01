from pixie_web import route, run, response, ProcessType

# Local synchronous

@route("/", ProcessType.local)
def index(env):
    return response("Hello world")

# Local async

@route("/async", ProcessType.local_async)
async def index_async(env):
    return response("Hello world (async)")

# Process-pooled (the default)

@route("/cpu")
def cpu_bound(env):
    from time import sleep
    sleep(3)
    return response("Hello world (CPU-bound operation)")

if __name__ == '__main__':
    run()