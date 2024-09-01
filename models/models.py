import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    model = models[model_spec['name']]()
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
