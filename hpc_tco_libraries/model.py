import json
import re
import numexpr  # for safe evaluation of arbitrary equations


def load_dataset(dataset_path):
    """
    Loads a dataset from JSON.
    """
    with open(dataset_path, "r") as dataset_file:
        return json.load(dataset_file)


def evaluate_cost(data, **kwargs):
    """
    Recursively evaluates the cost of an entry in the data.
    """

    if "breakdown" not in data:
        return float(data["value"])
    else:
        return numexpr.evaluate(data["value"].format(**{**kwargs, **{k: evaluate_cost(v, **kwargs) for k,v in data["breakdown"].items()}})).item()


re_variables = re.compile(r"\{([A-z0-9_ \-]+)}")
def validate_dataset(dataset, *args):
    """
    Checks for missing fields in the data.
    """
    formula_variables = set()
    breakdown_variables = set()

    def _validate(data):
        if "breakdown" not in data:
            try:
                float(data["value"])
            except ValueError as e:
                raise type(e)(f"Couldn't parse {data['value']} as a float. Search for and fix this value in the JSON file.")
            return True
        else:
            formula_variables.update(re_variables.findall(data["value"]))
            for variable, child in data["breakdown"].items():
                breakdown_variables.add(variable)
                _validate(child)

        return True

    missing_variables = formula_variables - breakdown_variables
    if missing_variables:
        raise ValueError(f"Values are missing for the following fields: {','.join(map(str, missing_variables))}")

    unused_variables = breakdown_variables - formula_variables
    if unused_variables:
        raise ValueError(f"The following fields are unused in a formula: {','.join(map(str, unused_variables))}")
    
    return all([_validate(entry) for entry in dataset.values()])