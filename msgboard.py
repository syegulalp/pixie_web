# Extremely primitive message board demo using `shelve`

import shelve

from pixie_web import route, run, template, response, ProcessType

main_template = template("post.html", "demo")

# because `shelve` is not threadsafe, we must run this demo single-threaded

@route("/", ProcessType.main)
def main(env):
    with shelve.open("db", flag="c", protocol=5) as db:

        if not "posts" in db:
            db["posts"] = ["Hi"]
            db.sync()

        data = db["posts"]

        if env.headers["_VERB"] == "POST":
            value = env.form["name"]
            data.append(value)
            if len(data) > 5:
                data.pop(0)
            db["posts"] = data

        text = main_template.format(
            "<ul>" + "".join([f"<li>{_}</li>" for _ in db["posts"]]) + "</ul>"
        )

    return response(text)


if __name__ == "__main__":
    run(port=80)
