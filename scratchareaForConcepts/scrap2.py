# http://stackoverflow.com/a/25292325
# describes callable objects.
import time
def timing_function(some_function):

    """
    Outputs the time a function takes
    to execute.
    """

    def wrapper(n):
        t1 = time.time()
        some_function(n)
        t2 = time.time()
        return "Time it took to run the function: " + str((t2 - t1)) + "\n"
    return wrapper


@timing_function
def my_function(n):
    num_list = []
    for num in (range(0, n)):
        num_list.append(num)
	print "Hello", num
    print "\nSum of all the numbers: " + str((sum(num_list))) 


print my_function(500)
