import yaml

data = dict(
    A = 'a',
    B = dict(
        E = [ [1,2,3,4,5], [2,3,4,5,6,7] ] ,
    )
)


# Write dictionary to file
with open('data.yml', 'w') as outfile:
    #outfile.write( yaml.dump(data, default_flow_style=True) )
    outfile.write( yaml.dump(data) )

# Read dictionary from file
with open('data.yml', 'r') as f:
    doc = yaml.load(f)

print doc
print doc['B']['E'][1][5] # Shoould give 7
