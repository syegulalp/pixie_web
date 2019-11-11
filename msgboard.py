# Extremely primitive message board demo using `shelve`
# because `shelve` is not threadsafe, we must run this demo single-threaded

import shelve
from pixie_web import route, run, Template, template, simple_response, RouteType, Unsafe

main_template = Template(None, "post.html", "demo")


def output(data, notice):
    msgs = Unsafe("".join(["<li>{}</li>".format(Unsafe(_).esc) for _ in data]))
    return simple_response(main_template.render(msgs, notice))


def s_open(flag):
    return shelve.open("db", flag=flag, protocol=5)


@route("/", RouteType.sync)
def main(env):
    with s_open("r") as db:
        data = db["posts"]
    return output(data, "")


@route("/", RouteType.sync, "POST")
def main_post(env):
    msg = ""
    with s_open("w") as db:
        data = db["posts"]
        value = env.form.get("text")
        if not value or value == "":
            msg = "You submitted a blank post. Be more creative."
        else:
            data.append(value)
            if len(data) > 5:
                data.pop(0)
            db["posts"] = data

    return output(data, msg)


if __name__ == "__main__":
    with s_open("c") as db:
        if not "posts" in db:
            db["posts"] = ["Hi"]
            db.sync()
    run(port=80, workers=None)
