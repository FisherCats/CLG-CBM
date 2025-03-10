def get_model(model_name, args):
    name = model_name.lower()
    if name == "playground":
        from models.clgcbm import Player
        return Player(args)
    else:
        assert 0
