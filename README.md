# `pixie_web`: a small, (hopefully) fast, `async`-powered web mini-framework for Python 3.8+

`pixie_web` (or just `pixie` for short) is a proof-of-concept mini-web framework for Python, a la [`bottle`](http://bottlepy.org/), that relies intentionally on the latest version of Python (3.8+) and its `async` idioms.

I wrote it mostly to teach myself `async` and modern Python, but if it seems like it could be genuinely useful, please feel free to submit pull requests.

`pixie_web`'s distinct feature is that it can satisfy requests on different routes using different processing mechanisms:

* Single-threaded, synchronous (in the main process)
* Single-threaded, asynchronous (in the main process)
* Multi-process, in a process pool

This way, you can quickly direct CPU-reliant traffic to a process pool, I/O-reliant traffic to an `async` pool, and so on. The amount of data sent to external process pools is kept to a minimum, to reduce the performance impact of serializing Python objects.

# Features

`pixie_web` is, again, proof-of-concept at this point, but it supports the following features:

* All core functions in a single Python file with no dependencies on anything but the standard library (although optional features might have third-party dependencies).
* Sync/async/process-pooled routing.
* Route decorators in the style of `bottle` or `flask`.
* Wildcard routes (e.g., `/route/<filename>`, with `filename` passed to the handler function as an argument).
* Load files from disk for use as simple templates.
* In-memory file and template caching, using Python's `functools.lru_cache` (you can require or suppress caching as needed).
* (In progress) Verb handlers for routes (`GET/POST`).
* (In progress) Form submissions, file uploads.

There are probably many bugs -- bad conformance to HTTP specs, etc.

Possible future features:

* Incremental/streaming responses by way of `yield`
* Better cross-process synchronicity features. The assumption right now is that processes in the pool have no shared state.
* A native, simple template engine (again, a la `bottle`).
* Cookies and some basic cookie security.
* Support for external server adapters (WSGI, etc.) instead of only the built-in server
* Before/after triggers for routes.
* ETags.
* Optional external library support:
  * `uvloop`
  * `httptools`
  * `ujson`
* HTTP/2, HTTP/3, HTTPS support.
* Tests!

Ideally the feature set would be small (again, a la `bottle`), but complete enough for the basics.

On the whole I'd rather support a few generally useful features well than a lot of them badly. For instance, there would be no native ORM; there's plenty of good third-pary ORMs out there.

# Example

```python
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
```

See `demo.py` for a more elaborate example and usage. (It's not very pretty but I'll work on that.)

# License

MIT