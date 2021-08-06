
def check(x, Type): 
    return x if isinstance(x, Type) else Type(x)

def primitive(Class):
    def register_methods(method):
        setattr(Class, method.__name__, method)
        return method 
    return register_methods


