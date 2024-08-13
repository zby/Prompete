from prompete import Chat
from pydantic import BaseModel
from pprint import pprint


class Address(BaseModel):
    street: str
    city: str


class Company(BaseModel):
    name: str
    speciality: str
    address: Address


class CompaniesList(BaseModel):
    companies_list: list[Company]


def print_companies(companies: CompaniesList):
    pprint(companies)
    return companies


# pprint(get_tool_defs([print_companies]))

file_path = 'examples/Three_Companies_Story.txt'
with open(file_path, 'r') as file:
    story = file.read()

prompt = f"{story}\n\nPlease print the information about companies mentioned in the text above."
# Create a Chat instance
chat = Chat(model="gpt-4o-mini")

reply_struct = chat(prompt, response_format=CompaniesList)

print_companies(reply_struct)

# OUTPUT
[
    Company(
        name='Aether Innovations',
        speciality='sustainable energy solutions',
        address=Address(street='150 Futura Plaza', city='Metropolis')),
    Company(
        name='Gastronauts',
        speciality='culinary startup',
        address=Address(street='45 Flavor Street', city='Metropolis')),
    Company(
        name='SereneScape',
        speciality='digital wellness',
        address=Address(street='800 Tranquil Trail', city='Metropolis'))]
