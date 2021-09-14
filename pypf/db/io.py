from pylab import append, csv2rec
import numpy
import re
import pdb

class dlmread:
    nfiles = 0 # count of number of textfile objects
    def __init__(self,fname,sep=None, hlines=0, dt=numpy.float, comment="#",
                 verbose=False):
        """
        Initialization function, which opens the specified file and returns
        the contents as a Numpy array

        Presently implemented for dt = str, where number of columns may vary from
        row to row. Rows will be filled with blank columns so that all the reported
        rows will have same number of columns.

        for dt = numpy.float, the method uses fromfile directly! No support for
        different number of columns for different rows.
        """
        dlmread.nfiles += 1
        self.name = fname # name
        self.sep = sep
        self.ncol = 0

        if dt == str:
            self.fh = open(fname) # handle for the file
            lst = self.fh.readlines()
            self.fh.close()

            self.header = []

            for l in range(hlines):
                # store the header lines in self.header
                self.header.append(lst.pop(0))
            #(self.header.append(lst.pop(0)) for l in range(hlines))

            self.getColNum(lst[0]) # get the number of columns in the file
            if self.ncol >0:
                # read the data into a Numpy arrray

                try:
                    self.data = []
                    self.comments = []
                    # routines to test regexp vs. str.split
                    # regexp approx 12 times slower! Use only when necessary

                    #import re
                    #
                    #def mystrsplit():
                    #     "28-11-2007 00:00;28-11-2007;-0.409;8000;2000;\'Data suspect due to unrealistic form with comparison to 25, 50 and 150cm; Data deleted due to sensor connection error\'".split(";")
                    #
                    #def myregexpsplit():
                    #     re.split(";(?=(?:[^\"]*\"[^\"]*\"|[^\']*\'[^\']*\')*(?![^\"]*\"|[^\']*\'))","28-11-2007 00:00;28-11-2007;-0.409;8000;2000;\'Data suspect due to unrealistic form with comparison to 25, 50 and 150cm; Data deleted due to sensor connection error\'")
                    #
                    #if __name__=='__main__':
                    #     from timeit import Timer
                    #     t1 = Timer("mystrsplit()", "from __main__ import mystrsplit")
                    #     t2 = Timer("myregexpsplit()", "from __main__ import myregexpsplit")
                    #     print t1.timeit()
                    #     print t2.timeit()

                    if self.sep != None:
                        reexp = self.sep + "(?=(?:[^\"]*\"[^\"]*\"|[^\']*\'[^\']*\')*(?![^\"]*\"|[^\']*\'))"

                    if len(lst) >= 1:
                        add_columns = False
                        for lid, line in enumerate(lst):
                            tmp = line.split(comment,1)   # Split line at first comment char only
                            if len(tmp) == 1:
                                self.comments.append("")
                            else:
                                self.comments.append(tmp[1])

                            line = tmp[0].rstrip()

                            if self.sep != None:
                                # First try to split the line regularly
                                line2 = line.split(self.sep)

                                # If number of columns not equal ncols do reg exp split
                                if len(line2) != self.ncol:
                                    if verbose:
                                        print("Line {0}, Trying re.split...".format(lid))
                                    line2 = re.split(reexp,line)
                                    # pdb.set_trace()
                                # If line is not empty...
                                if not line2 == ['']:
                                    if len(line2) > self.ncol:
                                        if verbose:
                                            print("Line {0}, len(line2) > self.ncol, updating ncol".format(lid))
                                        self.ncol = len(line2)
                                        add_columns = True
                                    elif len(line2) < self.ncol:
                                        if verbose:
                                            print("Line {0}, len(line2) < self.ncol, adding enpty entries".format(lid))
                                        # If line2 has less than ncol columns
                                        # add the missing columns as empty ("")
                                        #pdb.set_trace()
                                        [line2.append('') for lid in numpy.arange(self.ncol-len(line2))]

                                    self.data.append(line2)
                                    # pdb.set_trace()
                            else:
                                self.data.append(line)

                        if add_columns:
                            if verbose:
                                print("Updating number of columns in new pass...")
                            # If number of columns is not the same for all rows...
                            for line in self.data:
                                # if line has fewer columns than ncol
                                if len(line) < self.ncol:
                                    # If line has less than ncol columns
                                    # add the missing columns as empty ("")
                                    [line.append('') for cid in numpy.arange(self.ncol-len(line))]
                        #pdb.set_trace()
                        self.data = numpy.ma.array(self.data)
                    else:
                        self.data = None
                except:
                    print("Could not parse data file %s!" % fname)
                    pdb.set_trace()
            else:   # If zero columns found
                print("No data read!")
                self.data = []

        else:   # if dtype not specified as str
            if self.sep == None:
                self.sep = " "

            with open(fname) as self.fh: # handle for the file
                self.header = []

                #for l in range(hlines):
                #    # store the header lines in self.header
                #    self.header.append(self.fh.readline())
                (self.header.append(self.fh.readline()) for l in range(hlines))

                first_line = self.fh.readline()

            self.ncol = len(first_line.strip().split())


#            self.data = loadtxt(fname, dtype=dt, comments='#',
#                                delimiter=self.sep, converters=None,
#                                skiprows=hlines, usecols=None,
#                                unpack=False)

            with open(fname) as self.fh: # handle for the file
                #for l in range(hlines):
                #    # skip past header
                #    self.fh.readline()
                (self.fh.readline() for l in range(hlines))

                self.data = numpy.fromfile(self.fh, dtype=dt, count=-1, sep=self.sep)
                self.data = self.data.reshape(-1,self.ncol)





    def getColNum(self,line):
        """Function to find the number of columns in a dataset in a textfile.
        It parses the passed line, using the defined separator.

        NEW: takes into account text qualifiers " " or ' '
        """

        try:
            reexp = self.sep + "(?=(?:[^\"]*\"[^\"]*\"|[^\']*\'[^\']*\')*(?![^\"]*\"|[^\']*\'))"
            lineSplit = re.split(reexp,line)
            # lineSplit = line.split(self.sep)
            if self.sep == ' ':
                tmp = []
                for s in lineSplit:
                    if s != '': tmp = append(tmp, s)
                lineSplit = tmp
            self.ncol = len(lineSplit)
        except:
            print("Could not read number of columns from file")
            print("Assuming one column!")
            self.ncol = 1

def test():

    test_file = ['a,b,c','a,c,d','g,h,i']
    with open('tmpfile.txt', 'w') as f:
        f.write("\n".join(test_file))
    myfile1 = dlmread('tmpfile.txt',sep = ',', hlines = 0, dt = str)

    test_file = ['a,b,c','a,c,d,e','g,h,i']
    with open('tmpfile2.txt', 'w') as f:
        f.write("\n".join(test_file))
    myfile2 = dlmread('tmpfile2.txt',sep = ',', hlines = 0, dt = str)

    return myfile1, myfile2



def csv2dict(fname):
    a = csv2rec(fname)
    keys = list(a.dtype.fields.keys())
    data = {}
    for key in keys:
        data[key] = a[key]

    return data



