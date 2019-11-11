# Using Pixie

- [Using Pixie](#using-pixie)
- [Getting started](#getting-started)
- [A simple example](#a-simple-example)
- [Adding routes](#adding-routes)
- [`Request()` objects](#request-objects)
- [Route wildcards](#route-wildcards)
- [Route actions](#route-actions)
- [Async routes](#async-routes)
- [Process pools and pooled routes](#process-pools-and-pooled-routes)
- [Emitting responses](#emitting-responses)
  - [Send bytes](#send-bytes)
  - [Send a `simple_response()` object](#send-a-simple_response-object)
  - [Send a `Response()` object](#send-a-response-object)
- [Setting cookies](#setting-cookies)
- [WSGI](#wsgi)

# Getting started

`pixie_web` is self-contained. You can deploy it by simply inserting the file `pixie_web.py` into your project. The other files in the repository are there mostly for the sake of running the demos.

# A simple example

```python
from pixie_web import route, run

@route("/", RouteType.asnc) # sic `asnc`
async def index(request):
    return Response("Hello world")

if __name__ == '__main__':
    run()
```

Note that you need to have the `if __name__ == '__main__':` block at the bottom of the script to ensure `pixie_web` runs correctly.

# Adding routes

`pixie_web` uses the `@route` decorator to bind functions to routes on your site.

`@route` takes one argument by default, the route to bind to.

It also accepts a `route_type` argument that describes what kind of process isolation to apply to the route:

* `RouteType.sync`: Runs the route in the main thread of the default process. If you don't have any async routes or any thread-blocking operations, you can use this, but it's a good idea to use it only for the sake of transitioning older functions that aren't async-compatible. In short, if you're starting a new project, use `RouteType.async` or `RouteType.pool` (the default) instead.
* `RouteType.asnc`: Requires the route be an `async` function. Runs the function in the main process's default async loop. Most of the time you'll want to do this. Note that the name is `asnc`, not `async`, to avoid weird syntax issues.
* `RouteType.pool`: Runs the route in a process pool. This is best for long-running or CPU-intensive functions.
* `RouteType.stream`: Runs the route in a process pool, and allows returning results incrementally. Not recommended, but included to demonstrate how it's possible.

# `Request()` objects

Each route function automatically takes a `Request()` object as its first argument. This is a class that contains the details about the request made to the server.

* `Request.raw_data`: The entire request as originally sent to the server, stored as a `bytes` object. Most of the time you won't need to bother with this since all of the data stored in it is available in more elegant ways, but it's there if you need it.
* `Request.headers`: The headers for the request, stored as a Python dictionary.
* `Request.verb`: The GET, POST, or other action associated with the request.
* `Request.path`: The path for the request.
* `Request.protocol`: The HTTP protocol variation used for the request.
* `Request.body`: The body of the request, stored as `bytes`.
* `Request.form`: Any form data associated with the request, also stored as a Python dictionary.

# Route wildcards

You can indicate a wildcard route by using braces:

```python
@route("/file/<filename>", RouteType.sync)
def cached_routed_file(env, filename):
    return cached_file(filename, "demo")
```

The wildcards are passed to the route function in the order they are declared in the decorator.

# Route actions

By default all routes are associated only with the `GET` action. You can define which actions apply to which routes:

```python
@route("/", action=("GET", "POST"))
def index(env):
    return Response("Hello world")
```

# Async routes

Async routes use an async function and must be explicitly decorated to use them.

```python
@route("/async", RouteType.asnc)
async def index_async(env):
    return Response("Hello world (async)")
```

Note that as with any `async` routine, you should take care to not write blocking code. For that, use a process pool route.

# Process pools and pooled routes

Process pool routes farm out requests to a pool of Python subprocesses, so they can run independently without blocking the main request thread.

```python
@route("/cpu", RouteType.pool)
def cpu_bound(env):
    from time import sleep
    sleep(3)
    return Response("Hello world (CPU-bound)")
```

(Normally, if you wanted to sleep for 3 seconds without blocking anything, you'd use `async.sleep()`; we're just using the conventional `sleep()` function here to illustrate how this is a CPU-bound function.)

By default, `pixie_web` creates a process pool with one process per CPU in the host. To change this behavior, pass parameters to the `run()` method:

* `workers=True`: Use one process per CPU. (Default)
* `workers=False`: Do not use workers; just use one master process.
* `workers=4`: Pass an integer to create that many processes in the pool (in this case, 4).

# Emitting responses

`pixie_web` provides several possible ways to return responses to the client.

## Send bytes

This is the most basic way to send a response, but it means you're not emitting anything but what you specify manually -- headers are not automatically generated. To that end it's best to use the `header()` helper function first:

```python 
yield header(code=200) # 200 OK
yield b'Hello world'
```

It's often just easiest to use a `simple_response()` object. (See below.)

`Content-Length:` is not generated in the headers here, but you can add that manually to the `header()` function if you know it ahead of time.

```python
body = b'Hello world'
yield header(code=200, headers={'Content-Length':len(body)})
yield body
```

This is also the way to send a result incrementally if you use a streaming route.

## Send a `simple_response()` object

The `simple_response()` helper function generates headers and a body with minimal overhead, along with a `Content-Length` header. This is useful if you don't plan on modifying any of the data; you just want to send it on its way as quickly as possible.

```python
return simple_response("Hello world") # 200 OK is assumed
```

## Send a `Response()` object

The `Response()` class constructs an object to hold a response, which can be manipulated after the fact. This is a little slower due to the overhead of constructing the object, but most convenient.

```python
response = Response("Hello", code=200)
if some_condition:
    response.code=500
    response.body = "Oh dear, something blew up"
    response.header['X-Something-Happened'] = "Yes"
return response
```

`Response()` objects have a property, `Response.as_bytes()`, which yields up the byte stream that's actually sent to the client. You don't need to generate this; the server object does that if it encounters a `Response()` as a return from a route handler.

# Setting cookies

You can set cookies in a `Response()` object by simply passing a dictionary with the `cookies` keyword argument.

```python
@route("/async", RouteType.asnc)
async def index_async(env):
    return Response(
        "Hello world", cookies={"mycookie": 1},
    )
```    

# WSGI

WSGI support for `pixie_web` is still very primitive, but is possible:

```python
import pixie_web

@pixie_web.route("/", pixie_web.RouteType.sync)
def index(env):
    return Response("Hello world")

application = pixie_web.server.application
```

Current limitations of using WSGI: 

* `pixie_web` only properly supports `Response()` when using WSGI. `simple_response` and `bytes` with a `header()` support is in the works.
* You can only use `RouteType.sync`.

These limitations will eventually be removed or at least better compensated for.