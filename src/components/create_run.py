import mlflow
import dagshub


def run_id(name_experiment):

    dagshub.init(repo_owner='RiemanNClav', 
                 repo_name='-Default-of-Credit-Card-Clients-Dataset', mlflow=True)

    mlflow.set_experiment(name_experiment)

    run = mlflow.start_run()

    mlflow.end_run()

    return run.info.run_id


if __name__=="__main__":
    x = run_id('model_service')
    print(x)