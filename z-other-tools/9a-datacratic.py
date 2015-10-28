
## https://github.com/szilard/benchm-ml/issues/25

## from @nicolaskruchten, thanks :)

## This code gives an AUC of 0.7431 in 19.1s for the 1M training set on an r3.8xlarge EC2 instance 
## with the latest release of Datacratic's Machine Learning Database (MLDB), available at http://mldb.ai/



from pymldb import Connection
mldb = Connection("http://localhost/")

mldb.v1.datasets("bench-train-1m").put({
    "type": "text.csv.tabular",
    "params": { "dataFileUrl": "https://s3.amazonaws.com/benchm-ml--main/train-1m.csv" }
})

mldb.v1.datasets("bench-test").put({
    "type": "text.csv.tabular",
    "params": { "dataFileUrl": "https://s3.amazonaws.com/benchm-ml--main/test.csv" }
})

mldb.v1.procedures("benchmark").put({
    "type": "classifier.experiment",
    "params": {
        "experimentName": "benchm_ml",
        "training_dataset": {"id": "bench-train-1m"},
        "testing_dataset": {"id": "bench-test"},
        "configuration": {
            "type": "bagging",
            "num_bags": 100,
            "validation_split": 0,
            "weak_learner": {
                "type": "decision_tree",
                "max_depth": 20,
                "random_feature_propn": 0.3
            }
        },
        "modelFileUrlPattern": "file://tmp/models/benchml_$runid.cls",
        "label": "dep_delayed_15min = 'Y'",
        "select": "* EXCLUDING(dep_delayed_15min)",
        "mode": "boolean"
    }
})

import time

start_time = time.time()

result = mldb.v1.procedures("benchmark").runs.post({})

run_time = time.time() - start_time
auc = result.json()["status"]["folds"][0]["results"]["auc"]

print "\n\nAUC = %0.10f, time = %0.4f\n\n" % (auc, run_time)



