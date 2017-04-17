from functools import partial
import multiprocessing  
import sys
import traceback

class Streamer:
    
    ### PRIVATE FUNCTIONS
    def _builder(self, expression):
        self.STR = expression
        return self

    def _flatten(self, generator, function):
        for thing in generator:
            for value in function(thing):
                yield value

    def _take(self, generator, number):
        for x in range(number):
            yield next(generator)

    def _tap(self, generator, function):
        for x in generator:
            function(x)
            yield x
            
    def _pack(self, generator, size):
        arr = []
        for x in generator:
            arr.append(x)
            if len(arr) == size:
                ready = arr[:]
                arr = []
                yield ready
        if len(arr) is not 0:
            yield arr
            
    def _thread_cleanup(self):
        self.pool.close()
        self.pool.join()
        
    ##  OVERRIDDEN 
    def next(self):
        return next(self.STR)

    def __init__(self, iterable_element):
        self.STR = (x for x in iterable_element)
        try:
            cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            cpus = 2
        self.pool = multiprocessing.Pool(processes=(cpus))

    def __iter__(self):
        for x in self.STR:
            yield x
            
    def __next__(self): 
        return next(self.STR)

    ### TRANSFORMS
    def _map(self, function):
        return self._builder(map(function, self.STR))
    
    def _dmap(self, function):
        return self._builder(self.pool.imap_unordered(function, self.STR))

    def _flatmap(self, function):
        return self._builder(self._flatten(self.STR, function))
    
    ## use the following function with great caution! This loads all iterator data into memory for mapping.   
    def distributed_map(self, function, *args, **kwargs):
        return self._dmap(partial(function, *args,*kwargs))
    
    def map(self, function, *args, **kwargs):
        return self._map(partial(function, *args, **kwargs))

    def flatmap(self, function, *args, **kwargs):
        return self._flatmap(partial(function, *args, **kwargs))
    
    ### LIMITERS
    def filter(self, proposition):
        return self._builder((x for x in self.STR if proposition(x)))

    def take(self, number):
        return self._builder(self._take(iter(self.STR), number))

    ### RESHAPING  
    
    def pack(self, number):
        return self._builder(self._pack(self.STR,number))
    
    def distributed_pack(self):
        pass
    
    def flatten(self):
        return self._builder(self._flatten(self.STR, lambda x: x))
    
    ### PASSIVE
    def tap(self, function):
        return self._builder(self._tap(self.STR, function))
    
    ### CONSUMERS
    def print(self):        
        try:
            print(list(self.STR))
            self._thread_cleanup()
        except: 
            self._thread_cleanup()
            raise
        
        
    def consume(self, function):
        try:
            function(self.STR)
            self._thread_cleanup()
        except: 
            self._thread_cleanup()
            raise
        
    def reduce(self, function):
        try:
            ans = function(self.STR)
            self._thread_cleanup()
            return ans
        except: 
            self._thread_cleanup()
            raise
        
    def drain(self):
        try:
            for x in self.STR:
                pass
            self._thread_cleanup()
            return e
        except e: 
            self._thread_cleanup()
            raise e
        
    def distributed_red_map(self, function, factor = 2):
        _str = _stream(self.STR).pack(factor)        
        return self._builder(self.pool.imap_unordered(comb, _str))
    
