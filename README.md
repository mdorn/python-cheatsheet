## Magic/special methods

Common magic methods include:

- `__init__`
- `__str__`
- `__repr__`
- `__call__` - to turn instances of a class into callables (e.g. overload `__call__` to take a parameter)
- binary/comparison/etc. operators like `__add__`, `__gt__` etc.
- iterator methods like `__next__`
- `__getattr__`, `__setattr__` etc.

Examples of where you might want to overload/override them include:

- overriding the String representation on your object to be more readable with `__str__`
- overriding the `__add__` operator for a certain mathematical purpose e.g. Vector addition:

```
class Vector(object):
    def __init__(self, *args):
        """ Create a vector, example: v = Vector(1,2) """
        if len(args) == 0:
            self.values = (0,0)
        else:
            self.values = args
    def __add__(self, other):
        """ Returns the vector addition of self and other """
        added = tuple(a + b for a, b in zip(self.values, other.values) )
        return Vector(*added)
```

- or an `__add__` method to automatically translate metric and customary units

## Decorators

Example

```
def my_decorator(func):
    def func_wrapper(x):
        print("Before calling " + func.__name__ + " with " + x)
        func(x)
        print("After calling " + func.__name__)
    return func_wrapper

@my_decorator
def foo(x):
    print("Hi, foo has been called with " + str(x))
foo("foobar")
```

## Functional features

Functional programming (often abbreviated FP) is the process of building software by composing pure functions, avoiding shared state, mutable data, and side-effects . Functional programming is declarative rather than imperative, and application state flows through pure functions. Contrast with object oriented programming, where application state is usually shared and colocated with methods in objects. [Source](https://medium.com/javascript-scene/master-the-javascript-interview-what-is-functional-programming-7f218c68b3a0)

### Map

```
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
# [1, 4, 9, 16, 25]
```

### Reduce

# the following two are equivalent

```
product = 1
list = [1, 2, 3, 4]
for num in list:
    product = product * num
# product = 24

from functools import reduce
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])
# NOTE: in the first execution x is the first item in the list,
# y the second; in subsequent executions, x is the result of
# the given reduce function and y is the next item in the list
```

### Closures

A closure is the combination of a function bundled together (enclosed) with references to its surrounding state (the lexical environment). In other words, a closure gives you access to an outer function’s scope from an inner function. ... To use a closure, simply define a function inside another function and expose it. To expose a function, return it or pass it to another function. The inner function will have access to the variables in the outer function scope, even after the outer function has returned. [Source](https://medium.com/javascript-scene/master-the-javascript-interview-what-is-a-closure-b2f0d2152b36)

- commonly used to give objects data privacy... the enclosed variables are only in scope within the containing (outer) function
- currying: pass all of the arguments a function is expecting and get the result, or pass a subset of those arguments and get a function back that’s waiting for the rest of the arguments
- can use the outer function as a factory for creating functions that are somehow related
- basically, closures remember their context in which they were created, e.g. the values of the arguments of the creating function

## Object Orientation features

### Inheritance

An example:

```
class A(object):
    def __init__(self):
        self.x = 'foo'
        self.y = 'bar'
    def hello_world(self):
        print ('hello world: %s %s' % (self.x, self.y))

class B(A):
    def __init__(self):
        # super(B, self).__init__()  # python2
        super().__init__()  # python3
        self.x
        self.y = 'baz'

a = B()
a.hello_world()
# prints 'hello world: foo baz'
```

### Class decorators

- `@staticmethod` - used to group functions which have some logical connection with a class to the class
- `@classmethod` - one use is to define a "alternative constructor" e.g. x = MyClass() vs. x = MyClass.alt()
- `@property` - use if you need custom getter/setter functionality done on an attribute, e.g.:

```
class Percent(object):
    def __init__(self, val=0):
        self._decimal = val
    def calculate_percent(self):
        return str(self._decimal * 100) + '%'
    @property
    def decimal(self):
        print("Getting value")
        return self._decimal
    @decimal.setter
    def decimal(self, value):
        if value < 0:
            raise ValueError('Must be > 0')
        print("Setting value")
        self._decimal = value
```

## Concurrency/Parallelism [WORK IN PROGRESS]

Some links:

https://code.tutsplus.com/articles/introduction-to-parallel-and-concurrent-programming-in-python--cms-28612
https://www.toptal.com/python/beginners-guide-to-concurrency-and-parallelism-in-python
https://www.awsadvent.com/2016/12/04/exploring-concurrency-in-python-aws/

### The GIL

The Python interpreter is NOT FULLY THREAD-SAFE. In order to support multi-threaded Python programs, there’s a global lock, called the global interpreter lock or GIL, that must be held by the current thread before it can safely access Python objects.

In order to EMULATE CONCURRENCY OF EXECUTION, the interpreter regularly tries to switch threads (see sys.setswitchinterval()). The lock is also released around potentially blocking I/O operations like reading or writing a file, so that other Python threads can run in the meantime.  A piece of code is thread-safe if it functions correctly during simultaneous execution by multiple threads. In particular, it must satisfy the need for multiple threads to access the same shared data, and the need for a shared piece of data to be accessed by only one thread at any given time.

[Source](https://docs.python.org/3/c-api/init.html)

NOTES:

- For IO bound tasks (e.g. network access, writing to disk), threads can be OK; for CPU-bound tasks, you'll want to take advantage of all CPU cores by using multiple processes instead of threads
- A thread can only modify Python data structures if it holds this global lock
- If you use threads to make a GUI responsive or to run other code during blocking I/O, the GIL won't affect you. If you use threads to run code in some C extension like NumPy/SciPy concurrently on multiple processor cores, the GIL won't affect you either.

### Threads

- Threading is a feature usually provided by the operating system. Threads are lighter than processes, and share the same memory space.

#### threading module (old)

```
import logging
import time
from queue import Queue
from threading import Thread
logging.basicConfig(level='INFO')
NUM_THREADS = 8
def job(num):
    logging.info('Starting my job: {}'.format(num))
    time.sleep(2)
    logging.info('Finished my job: {}'.format(num))
class Worker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue
    def run(self):
        while True:
            num = self.queue.get()
            try:
                job(num)
            finally:
                self.queue.task_done()
def main():
    queue = Queue()
    for x in range(NUM_THREADS):
        worker = Worker(queue)
        worker.daemon = True  # let main thread exit even tho workers block
        worker.start()
    for i in range(10):
        logging.info('Queueing {}'.format(i))
        queue.put(i)
    queue.join()  # cause main thread to wait for queue to finish all jobs
if __name__ == '__main__':
    main()
```

#### concurrent.futures.ThreadPoolExecutor (new since 3.2)

```
import logging
import time
from concurrent.futures import ThreadPoolExecutor, wait
NUM_WORKERS=5
logging.basicConfig(level=logging.INFO)
def job(num):
    logging.info('Starting my job: {}'.format(num))
    time.sleep(2)
    logging.info('Finished my job: {}'.format(num))
def main():
    args = range(10)
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(job, args, timeout=30)
    # with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     futures = {executor.submit(job, arg) for arg in args}
    #     wait(futures)
if __name__ == '__main__':
    main()
```

### Multiprocessing

- true parallelism, but it comes with a cost. The entire memory of the script is copied into each subprocess that is spawned

```
import logging
from multiprocessing.pool import Pool
logging.basicConfig(level=logging.INFO)
def job(num):
    logging.info('Starting my job: {}'.format(num))
    time.sleep(2)
    logging.info('Finished my job: {}'.format(num))
def main():
    args = range(10)
    with Pool(4) as p:
        p.map(job, args)
if __name__ == '__main__':
    main()
```

### Distributed message queue

For distributing to other cores -- below is using Python RQ: http://python-rq.org  but could also use Celery or other.

```
redis-server /usr/local/etc/redis.conf
# QUEUE
from redis import Redis
from rq import Queue
q = Queue(connection=Redis(host='localhost', port=6379))
    for i in range(10):
        q.enqueue(job, num)
# WORKERS
from rq import Connection, Worker
with Connection():
    w = Worker('default')
    w.work()
```

### Async/Await (Python 3.5+ only)

TODO -- asyncio module, see also aiohttp https://aiohttp.readthedocs.io/en/stable/index.html

## OTHER

- PROFILING:
	- `import cProfile;cProfile.run('my_func(x)')`
	- `import timeit;print (timeit.timeit('check_prime(10000169)', setup='from __main__ import check_prime', number=1))`
- DEBUGGING: `import pdb;pdb.set_trace()`
- MANUAL COMPILATION (with optimization to strip assertions, docstrings, etc.): `python -OO -m compileall /path/to/your/files`
- REGEX:
  - `re.findall(A, B)` | Matches all instances of an expression A in a string B and returns them in a list.
  - `re.search(A, B)` | Matches the first instance of an expression A in a string B, and returns it as a re match object.
  - `re.split(A, B)` | Split a string B into a list using the delimiter A.
  - `re.sub(A, B, C)` | Replace A with B in the string C
- LOGGING:

```
import logging
logging.basicConfig(level=logging.INFO)
logging.info('Here's a message')
```

- STRING FORMATTING:
  - `'hello {1} {0}'.format('foo', 'bar')  # 'hello bar foo'`
  - `'hello {foo} {bar}'.format(foo=1, bar=2)  # 'hello 1 2'`
  - `'hello {foo} {bar}'.format(**{'foo': 1, 'bar': 2})  # 'hello 1 2'`
  - many other options: https://pyformat.info/
- ROUNDING/FORMATTING NUMBERS
  - `x = float(1) / float(3)  # 0.3333333333333333`
  - `round(x, 2)  # 0.33`
  - `'{:.2f}'.format(x)  # '0.33' -- a string; the old way was like this: '%.2f' % x`
  - use `Decimal` class for proper decimal arithmetic:

```
>>> Decimal(1) / Decimal(7)
Decimal('0.142857')
```

- DATE/TIME STUFF:
  - `datetime.now()  # datetime object for current time`
  - `time.time()  # produce Unix timestamp from current time`
  - `datetime.utcfromtimestamp(time.time()) # convert a Unix timestamp into a date object: UTC`
  - `five_min_ago = datetime.now() - timedelta(minutes=5)`
  - `datetime.strptime()`
  - `dateutil.parser.parse(datestring)  # parse ISO date string into date object  (THIRD PARTY)`
  - also see ARROW library
- VIRTUALENV:
  - `python3 -m venv env`
- SIMPLE HTTP SERVER:
	- `python -m SimpleHTTPServer 3000`

## TODO:

- Flask basics
- Some useful short examples:
	- save an image from a URL
	- scrape data from IMDB (e.g. using requests and BeautifulSoup)
