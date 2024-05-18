from giza_actions.action import Action, action
from giza_actions.task import task

@task
def print_hello():
    print(f"Hello Action!")

@action
def hello_world():
    print_hello()

if __name__ == '__main__':
    action_deploy = Action(entrypoint=hello_world, name="hello-world-action")
    action_deploy.serve(name="hello-world-action-deployment")