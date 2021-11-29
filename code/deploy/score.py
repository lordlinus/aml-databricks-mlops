import os

import joblib
import numpy as np
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.schema_decorators import input_schema, output_schema

from azureml.core.model import Model


# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model

    try:
        # The AZUREML_MODEL_DIR environment variable indicates
        # a directory containing the model file you registered.
        model_filename = "model.pkl"
        model_path = os.path.join(
            os.getenv("AZUREML_MODEL_DIR"), model_filename
        )
        print(model_path)
        model = joblib.load(model_path)
    except Exception as e:
        print("Failed to load model: " + str(e))
        pass


# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.
# sample json input: {"data": [[0.1, 1.2, 2.3, 3.4, 4.5]]}
@input_schema("data", NumpyParameterType(np.array([[0.1, 1.2, 2.3, 3.4, 4.5]])))
@output_schema(NumpyParameterType(np.array([0.1, 1.2])))
def run(data):
    # Use the model object loaded by init().
    result = model.predict(data)

    # You can return any JSON-serializable object.
    return result.tolist()
