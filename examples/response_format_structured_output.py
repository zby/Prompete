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

    def print_companies(self):
        pprint(self.companies_list, indent=4)


# pprint(get_tool_defs([print_companies]))

file_path = "examples/Three_Companies_Story.txt"
with open(file_path, "r") as file:
    story = file.read()

# model = "claude-3-haiku-20240307"  # Currently Anthropic does not support response_format - but we emulate it by using tools
model = "gpt-4o-mini"
chat = Chat(model=model)

prompt = f"{story}\n\nPlease print the information about companies mentioned in the text above."

reply_struct = chat(prompt, response_format=CompaniesList)

reply_struct.print_companies()

# OUTPUT
[
    Company(
        name="Aether Innovations",
        speciality="Sustainable energy solutions",
        address=Address(street="150 Futura Plaza", city="Metropolis"),
    ),
    Company(
        name="Gastronauts",
        speciality="Urban dining experiences",
        address=Address(street="45 Flavor Street", city="Gourmet Grove"),
    ),
    Company(
        name="SereneScape",
        speciality="Digital wellness",
        address=Address(street="800 Tranquil Trail", city="Metropolis"),
    ),
]
