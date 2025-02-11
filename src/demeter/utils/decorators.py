import functools
import inspect
import warnings
from time import time
from .toolbox import  format_time

def time_it(func):
    """
    This decorator is used to measure the execution time  (in seconds) of a function

    Usage example :
    .. code-block:: python
        import time
        @time_it
        def my_function():
            print("I fall alseeep ...)
            time.sleep(2.5)
            print("Hello world")

        my_function()
    """
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"\nComputation of {func.__name__} done in ",format_time(t2 -t1)," s")
        return result
    return wrap_func

string_types = (type(b''), type(u''))


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    source : https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically

    Usage example :

    .. code-block:: python

        @deprecated("use another function")
        def some_old_function(x, y):
            return x + y

        class SomeClass(object):
            @deprecated("use another method")
            def some_old_method(self, x, y):
                return x + y

        @deprecated("use another class")
        class SomeOldClass(object):
            pass


        some_old_function(5, 3)
        SomeClass().some_old_method(8, 9)
        SomeOldClass()
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))
