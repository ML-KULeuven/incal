def timeout(func, args=None, kwargs=None, duration=1, default=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = dict()

    if duration is None:
        return func(*args, **kwargs)

    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(duration)

    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        result = default
    finally:
        signal.alarm(0)

    return result
