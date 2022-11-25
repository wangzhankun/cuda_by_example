#%%
import os

# %%

dirs = [i for i in os.listdir() if i.find("chapter") == 0]

dirs.append('appendix_a')

# %%

for i in dirs:
    target = ''
    cus = [cu for cu in os.listdir(i) if cu.endswith(".cu")]
    for cu in cus:
        target += 'target("' + cu[:-3] + '")\n'
        # target += '\tset_objectdir("$(buildir)/' + i + '")\n'
        target += '\tset_kind("binary")\n'
        target += '\tadd_files("'+ cu + '")\n'
        target += 'target_end()\n\n\n'

    with open(os.path.join(i,'xmake.lua'), 'w', encoding='utf8') as f:
        f.write(target)

    

# %%
