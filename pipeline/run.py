from datetime import datetime
from glob import glob
from pathlib import Path
from ploomber.spec import DAGSpec


def now():
    return str(datetime.now().strftime("%Y-%m-%d-%H-%M"))

if __name__ == '__main__':
    dag = DAGSpec('pipeline/pipeline.yaml', env={'execution_time': now()}).to_dag()
    dag.build()