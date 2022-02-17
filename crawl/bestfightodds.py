import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import numpy as np

url = "https://www.bestfightodds.com/search?query={name}"

def get_fighter_link_by_name(name):
    page = requests.get(url.format(name=name))
    soup = BeautifulSoup(page.content, 'html.parser')
    a_list = soup.find_all("a", string=name)
    if len(a_list) >= 1:
        link = a_list[0]['href']
        return link.rsplit('/', 1)[-1]
    else:
        h1_list = soup.find_all('h1', id='team-name', string=name)
        if len(h1_list) >= 1:
            link = page.url
            return link.rsplit('/', 1)[-1]

    print(name)
    return None

def name_to_id(name):
    return name.lower().replace('\'','').replace('.','').replace(' ', '')

df = pd.read_csv('../data/raw/total_fight_data.csv', delimiter=';')
df.R_fighter = df.R_fighter.str.strip()
df.B_fighter = df.B_fighter.str.strip()

df['rid'] = df.R_fighter.apply(lambda n: name_to_id(n))
df['bid'] = df.B_fighter.apply(lambda n: name_to_id(n))
# df['rid'] = df.R_fighter.str.lower()
# df['rid'] = df.R_fighter.str.replace('\'','')
# df['rid'] = df.R_fighter.str.replace('.','')
# df['rid'] = df['rid'].str.replace(" ", "")
# .replace('\'','').replace(' ', '')

alias = {
    "Aoriqileng": "Qileng Aori",
    "Jacare Souza": "Ronaldo Souza"
}


# df.replace({"TJ Brown":"T.J. Brown"}, inplace = True)
# df.replace({"Don'Tale Mayes":"Don'tale Mayes"}, inplace = True)
# df.replace({"Aoriqileng":"Qileng Aori"}, inplace = True)
# df.replace({"Alatengheili":"Heili Alateng"}, inplace = True)
# df.replace({"Sumudaerji":"Su Mudaerji"}, inplace = True)
# df.replace({"Alex Da Silva":"Alex da Silva"}, inplace = True)
# df.replace({"MacDonald":"Macdonald"}, inplace = True)
# df.replace({"Jacare Souza":"Ronaldo Souza"}, inplace = True)
# df.replace({"JP Buys":"J.P. Buys"}, inplace = True)
# df.replace({"JJ Aldrich":"J.J. Aldrich"}, inplace = True)
# df.replace({"JC Cottrell":"J.C. Cottrell"}, inplace = True)
# df.replace({"Guangyou Ning":"Ning Guangyou"}, inplace = True)
# df.replace({"BJ Penn":"B.J. Penn"}, inplace = True)
# df.replace({"Zhang Tiequan":"Tiequan Zhang"}, inplace = True)
# df.replace({"Bibulatov Magomed":"Magomed Bibulatov"}, inplace = True)
# df.replace({"Joshua Burkman":"Josh Burkman"}, inplace = True)



names = np.concatenate([df.R_fighter.values, df.B_fighter.values])
names_unique = np.unique(names)

for n in names_unique:
    l = n.split()
    if (len(l) < 2):
        print(n)

names = []
links = []
for n in names_unique:
    l = get_fighter_link_by_name(n)
    if l:
        names.append(n)
        links.append(l)

    time.sleep(0.1)

df_links = pd.DataFrame(columns = ['name', 'link'])
df_links['name'] = names
df_links['link'] = links

print(df_links.head())
print(df_links.shape)
df_links.to_csv('../data/raw/fighter_links.csv', index = False)

# r_links = []
# b_links = []
# for index, row in df.iterrows():
#     r_name = row['R_fighter']
#     b_name = row['B_fighter']
#
#     r_link = get_fighter_link_by_name(r_name)
#     time.sleep(0.2)
#     b_link = get_fighter_link_by_name(b_name)
#
#     r_links.append(r_link)
#     b_links.append(b_link)
#
# # T.J. Brown
#
# # fighters/Don-tale-Mayes-6830
# print()
# df[]