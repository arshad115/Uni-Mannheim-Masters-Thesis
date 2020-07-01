def inplace_change(filename,fixed, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename, encoding="utf8") as original:
        # Safely write the changed content, if found in the file
        with open(fixed, 'a', encoding="utf8") as f:
            for line in original:
                print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
                s = line.replace(old_string, new_string)
                f.write(s)


inplace_change("data\webisalod-instances.nq","data\webisalod-instances_fixed.nq", "wasQuotedFrom> <", "wasQuotedFrom> <http://")