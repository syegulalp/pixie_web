# Extremely primitive message board demo using `shelve`

import shelve

from pixie_web import route, run, template, response, RouteType

main_template = template("post.html", "demo")

# because `shelve` is not threadsafe, we must run this demo single-threaded


@route("/", RouteType.sync)
def main(env):

    msg = ""

    with shelve.open("db", flag="c", protocol=5) as db:

        if not "posts" in db:
            db["posts"] = ["Hi"]
            db.sync()

        data = db["posts"]

        if env.verb == "POST":
            value = env.form["text"]
            if value != "":
                data.append(value)
                if len(data) > 5:
                    data.pop(0)
                db["posts"] = data
            else:
                msg = "<p>You submitted a blank post. Be more creative.</p>"

        text = main_template.format(
            "<ul>" + "".join([f"<li>{_}</li>" for _ in data]) + "</ul>", msg
        )

    return response(text)


if __name__ == "__main__":
    run(port=80, workers=None)
