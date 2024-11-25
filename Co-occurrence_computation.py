from collections import Counter
import pandas as pd
from glob import glob
import itertools 
import time


def parseLine(tree, tree_number):
    """This function responsibles of parse the lines according to ,

    Parameters
    ----------
    tree : str
        string which describes the instances' distribution in some tress
    tree_number : int
        
    Returns
    -------
    list
          list of pairs : <key =  tree number + leaf number , value = instance number>
    """
    parts = tree.split(',')
    ans = []
    for i in range(0, len(parts)):
      ans.append(((tree_number,parts[i]), i)) 

    return ans



def combinations(row):
    """ This function return all the possible options of combinations

    Parameters
    ----------
    row : str
        string which describes instances that felt in the same leaf
    tree_number : int
       
    Returns
    -------
    list
         list of edges (pairs of instances)
    """
    l = sorted(row)
    return itertools.combinations(l, 2)
  
  
def addToEdge(edge):
    """ This function cleans the edges from some symbols and adds some symbols that are required later.

    Parameters
    ----------
    edge : str
        string (instance1, instance2)
       
    Returns
    
    str
        instance1 instance2 0 |{}|
    """
    e = str(edge).replace("," ,"")
    e = e.replace("(" ,"")
    e = e.replace(")", "")
    return e + " 0 |{}|"


def flatten(inp):
    for i in inp:
        for j in i:
            yield j       


def main(list_paths, datasets_names_list, output1, output2):
    """ This function computes the co- co-occurrence score for each pair of instacnes.
    Then , removes the pairs with the weakest co-occurence scores (10% ) 
    The others pairs, are written to two different files: output1, output2

    Parameters
    ----------
    list_paths : list
        list of the datasets' paths
    datasets_names_list: list
        The datasets's names, orderd like the list_paths
    output1: str
       Path to the first output file
    output2: str
       Path to the second output file
    
    """

    list_paths=list_paths
    list_names = datasets_names_list
    
    dataset_names_dic= {}
    for num in range(0, len(list_names)):
      dataset_names_dic[num] = list_names[num]

    counter_data_set =0
    
  
    for path in list_paths:
        print(path)
      
        dataset_name = dataset_names_dic[counter_data_set]
        counter_data_set = counter_data_set +1
        
        # Load and parse data file into an RDD of LabeledPoint
        with open(path) as f:
            data = f.readlines()
        temp = []
        counter = 1

        for i in data:
          instances_number = len(i.split(",")) # Find the instances number 
          temp.append((i , counter))  # Add line number for each line
          counter = counter +1

        data = temp

        a = flatten(map(lambda tp :parseLine(tp[0],tp[1]), data))
        temp_a = {}
        for group, values in a:
            if group in temp_a:
                temp_a[group].append(values)
            else:
                temp_a[group] = [values]

        a = temp_a

        selected_edge = Counter(flatten(map(lambda x: combinations(x), a.values()))).items()
        del temp_a
        selected_edge = sorted(selected_edge, key=lambda x: -x[1])
        edges_amount_before_filter = len(list(selected_edge))
        precentage = 0.9
        wish_number_of_edges = edges_amount_before_filter*precentage

        d = selected_edge[: int(wish_number_of_edges)]
        d = [x[0] for x in d]

        gw_begin =""
        for i in range(0, instances_number): 
          gw_begin = gw_begin + "|{}|" +"\n" # Save according to 
        edge_gw = [addToEdge(edge) for edge in d]

        s_gw = str(list(edge_gw))
        s_gw = s_gw.replace(" '" , "")
        s_gw = s_gw.replace("'" , "")
        s_gw = s_gw.replace("," , "\n")
        s_gw = s_gw.replace("[","")
        s_gw = s_gw.replace("]","")
      
        s_with_count = "" + str(instances_number) + " " + str(len(list(d)))+"\n" # string that save edges list with the number of appearence

        s = str(list(d))

        s = s.replace(")," ,"\n")
        s = s.replace(" (", "")
        s = s.replace(",", "")
        s = s.replace("[","")
        s = s.replace("]","")
        s = s.replace("(", "")
        s = s.replace(")", "")
        
        with open(output1 + dataset_name + ".gw", "w") as f:
            f.write(gw_begin + s_gw)
        with open(output2 + dataset_name + ".in", "w") as f:
            f.write(s_with_count + s)
        del data
        
        # The output files are required later by the graphlet correlation distance method
        # The format of Output1:
          #LEDA graph format
        
        # The format of output2 (.in file):
          #Input file describes the network in a simple text format. 
          #The first line contains two integers n and e - the number of nodes and edges. 
          #The following e lines describe undirected edges with space-separated ids of their endpoints. 
          #Node ids should be between 0 and n-1 (see example.in).
        

if __name__ == "__main__":
    paths = sorted(glob("datasets-classified/*.csv"))
    names = [path.replace("datasets-classified/", "").replace(".csv", "") for path in paths]
    main(paths, names, "output1/", "output2/")
