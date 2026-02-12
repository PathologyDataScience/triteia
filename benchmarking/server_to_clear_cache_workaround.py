"""
A simple Flask application exposing an endpoint to execute a system command.

The motivation for this script is docker: a container user typically does not
have system privileges. So running "sudo -S sysctl -w vm.drop_caches=3" will not work.
However,this script can be run outside of docker as a privileged user. As long as
a container has HTTP access, it can remotely trigger this script,
which will be run as the privileged user.

Do *not* use this script in production or in an environment with open ports.
"""

from flask import Flask, Response
import subprocess

app = Flask(__name__)


@app.route("/run", methods=["GET", "POST"])
def run():
    # Run the command and wait for it to finish
    result = subprocess.run(["sudo", "-S", "sysctl", "-w", "vm.drop_caches=3"])

    # Only return 200 once the command has completed
    if result.returncode == 0:
        return Response("Command finished\n", status=200)
    else:
        return Response("Command failed\n", status=500)


if __name__ == "__main__":
    print("running on port 7987")
    app.run(host="127.0.0.1", port=7987)
