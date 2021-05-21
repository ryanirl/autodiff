
def check(x, Type): 
    return x if isinstance(x, Type) else Type(x)

def topological_sort(tensor):
    topologically_sorted_DAG = []
    visited = set()

    def recurse(tensor):
        if tensor not in visited:
            visited.add(tensor)

            for child in tensor._children:
               recurse(child)

            topologically_sorted_DAG.append(tensor)

    recurse(tensor)

    return topologically_sorted_DAG

def primitive(Class):
    def register_methods(method):
        setattr(Class, method.__name__, method)
        return method 
    return register_methods



