import mlflow
import os
import dagshub
# from dotenv import load_dotenv


def run_id(name_experiment):

    dagshub.init(repo_owner='RiemanNClav', repo_name='Traffic-Alert-CDMX-Chatbot', mlflow=True)

    mlflow.set_experiment(name_experiment)

    run = mlflow.start_run()

    mlflow.end_run()

    return run.info.run_id


if __name__=="__main__":
    x = run_id('model_service')
    print(x)