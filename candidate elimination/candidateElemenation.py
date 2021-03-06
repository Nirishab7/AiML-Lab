import csv

def min_generalizations(h, x):
    h_new = list(h)
    for i in range(len(h)):
        if not more_general(h[i:i+1],x[i:i+1]):
            h_new[i] = '?' if h[i] != '0' else x[i]
    return [tuple(h_new)]


def min_specializations(h, domains, x):
    results = []
    for i in range(len(h)):
        if h[i] == "?":
            for val in domains[i]:
                if x[i] != val:
                    h_new = h[:i] + (val,) + h[i+1:]
                    results.append(h_new)
        elif h[i] != "0":
            h_new = h[:i] + ('0',) + h[i+1:]
            results.append(h_new)
    return results



def generalize_S(x, G, S):
    S_prev = list(S)
    for s in S_prev:
        if s not in S:
            continue
        if not more_general(s,x):
            S.remove(s)
            Splus = min_generalizations(s, x)
            ## keep only generalizations that have a counterpart in G
            S.update([h for h in Splus if any([more_general(g,h) 
                                               for g in G])])
            ## remove hypotheses less specific than any other in S
            S.difference_update([h for h in S if 
                                 any([more_general(h, h1) 
                                      for h1 in S if h != h1])])
    return S

def specialize_G(x, domains, G, S):
    G_prev = list(G)
    for g in G_prev:
        if g not in G:
            continue
        if more_general(g,x):
            G.remove(g)
            Gminus = min_specializations(g, domains, x)
            ## keep only specializations that have a conuterpart in S
            G.update([h for h in Gminus if any([more_general(h, s)
                                                for s in S])])
            ## remove hypotheses less general than any other in G
            G.difference_update([h for h in G if 
                                 any([more_general(g1, h) 
                                      for g1 in G if h != g1])])
    return G

def more_general(h1, h2):
    #print(".....",h1,h2)
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == "?" or (x != "0" and (x == y or y == "0"))
        more_general_parts.append(mg)
    return all(more_general_parts)

def get_domains(examples):
    d = [set() for i in examples[0]]
    for x in examples:
        for i, xi in enumerate(x):
            d[i].add(xi)
    return [list(x) for x in d]

def candidate_elimination(examples):
    domains = get_domains(examples)[:-1]
    #print(domains)    #domains holds all possible values in a each column   ;   list of list 
    #[:-1] removes ['Y','N']
    #domains=[['japan', 'usa'], ['toyota', 'honda', 'chrysler'], ['white', 'red', 'blue', 'green'], ['1970', '1990', '1980'], ['economy', 'sports']]
    
    G = set([("?",)*(len(domains))])
    S = set([('0',)*(len(domains))])
    #print(G)

    i=0
    print("\n G[{0}]: {1}".format(i,G))
    print("\n S[{0}]:".format(i),S)
    for instance in examples:
        i=i+1
        x, label = instance[:-1], instance[-1]  # Splitting data into attributes and decisions

        if label=='Y': # x is positive example
            G = {g for g in G if more_general(g, x)}   
            #print(G)
            S = generalize_S(x, G, S)

        else: # x is negative example
            S = {s for s in S if not more_general(s,x)}
            G = specialize_G(x, domains, G, S)
        print("\n G[{0}]:".format(i),G)
        print("\n S[{0}]:".format(i),S)
    return 

data=csv.reader(open("candidate elimination/car.csv"))
examples=[]
for x in data:
    examples.append(tuple(x))   #List of  tuples
#print(examples)
candidate_elimination(examples)
