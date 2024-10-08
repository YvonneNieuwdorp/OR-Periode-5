import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

Orders = pd.read_excel('PaintShop - September 2024.xlsx', sheet_name='Orders')
Machines = pd.read_excel('PaintShop - September 2024.xlsx', sheet_name='Machines')
Setups = pd.read_excel('PaintShop - September 2024.xlsx', sheet_name='Setups')

current_time_m1 = 0
current_time_m2 = 0
current_time_m3 = 0
scheduled_orders = []
available_orders = Orders.copy()
current_colour_m1 = None
current_colour_m2 = None
current_colour_m3 = None

def processing_time(ordernumber, number_of_machine):
    """ Functie die de processing time berekent
    Parameters: 
        - ordernummer, als een int
        - machine die gebruikt wordt, als een int
    Output: de tijd die nodig is om de order uit te voeren/processingtime
    """
    return Orders.loc[ordernumber, 'Surface']/ Machines.loc[number_of_machine, 'Speed']

def setup_time(prev_colour, new_colour):
    """ geeft de setuptime aan
    Parameters
        - previous color als een string
        - new color als een string
    Output: setup_time
    """
    result = Setups.loc[(Setups['From colour'] == prev_colour) & (Setups['To colour'] == new_colour), 'Setup time']
    if not result.empty:
        return result.values[0]
    else:
        return 0 

def nearest_neighbor(current_time_m1, current_time_m2,current_time_m3, available_orders, current_colour_m1, current_colour_m2,current_colour_m3):
    """ Berekent de oplossing die het dichtbij het meest optimaal is, gebaseerd op minimale kosten
    Parameters
        - current_time: huidige tijd
        - available_order: orders die nog beschikbaar zijn, niet uitgevoerd
        - current_colour: huidige kleur
    Output: order die in huidige reeks het meest optimaal is
    """
    best_order = None
    best_machine = None
    min_cost = float('inf')

    for index, order in available_orders.iterrows(): 
        time_to_finish_m1 = current_time_m1 + processing_time(index, 0)
        lateness_m1 = max(0, time_to_finish_m1 - order['Deadline'])
        cost_m1 = lateness_m1 * order['Penalty']
        cost_m1 += setup_time(current_colour_m1, order["Colour"]) if current_colour_m1 else 0

        time_to_finish_m2 = current_time_m2 + processing_time(index, 1)
        lateness_m2 = max(0, time_to_finish_m2 - order['Deadline'])
        cost_m2 = lateness_m2 * order['Penalty']
        cost_m2 += setup_time(current_colour_m2, order["Colour"]) if current_colour_m2 else 0

        time_to_finish_m3 = current_time_m3 + processing_time(index, 2)
        lateness_m3 = max(0, time_to_finish_m3 - order['Deadline'])
        cost_m3 = lateness_m3 * order['Penalty']
        cost_m3 += setup_time(current_colour_m3, order["Colour"]) if current_colour_m3 else 0
        
        if cost_m1 < min_cost:
            min_cost = cost_m1
            best_order = index
            best_machine = "M1"
        if cost_m2 < min_cost:
            min_cost = cost_m2
            best_order = index
            best_machine = "M2"
        if cost_m3 < min_cost:
            min_cost = cost_m3
            best_order = index
            best_machine = "M3"
    return best_order, best_machine

# Lijst om iteratieresultaten op te slaan
results = []

# Scheduling loop for both machines
while not available_orders.empty:
    next_order_idx, best_machine = nearest_neighbor(current_time_m1, current_time_m2, current_time_m3, available_orders, current_colour_m1, current_colour_m2, current_colour_m3)
    
    if next_order_idx is not None:
        next_order = available_orders.loc[next_order_idx]
        
        # Bereken de starttijd, verwerkingstijd en setuptijd
        if best_machine == "M1":
            setup = setup_time(current_colour_m1, next_order["Colour"])
            processing = processing_time(next_order_idx, 0)
            start_time = current_time_m1 + setup
            finish_time = start_time + processing
            lateness = max(0, finish_time - next_order['Deadline'])
            current_time_m1 = finish_time
            current_colour_m1 = next_order["Colour"]

        elif best_machine == "M2":
            setup = setup_time(current_colour_m2, next_order["Colour"])
            processing = processing_time(next_order_idx, 1)
            start_time = current_time_m2 + setup
            finish_time = start_time + processing
            lateness = max(0, finish_time - next_order['Deadline'])
            current_time_m2 = finish_time
            current_colour_m2 = next_order["Colour"]

        else:  # best_machine == "M3"
            setup = setup_time(current_colour_m3, next_order["Colour"])
            processing = processing_time(next_order_idx, 2)
            start_time = current_time_m3 + setup
            finish_time = start_time + processing
            lateness = max(0, finish_time - next_order['Deadline'])
            current_time_m3 = finish_time
            current_colour_m3 = next_order["Colour"]

        # Voeg de huidige status van de planning toe aan de resultaten
        results.append({
            "Order": next_order['Order'],
            "Machine": best_machine,
            "Start Time": start_time,
            "Finish Time": finish_time,
            "Colour": next_order['Colour'],
            "Setup Time": setup,
            "Processing Time": processing,
            "Lateness": lateness,
            "Penalty Cost": lateness * next_order['Penalty']
        })

        # Verwijder de geplande order uit de beschikbare orders
        available_orders.drop(next_order_idx, inplace=True)

# Zet resultaten om in een DataFrame voor verdere verwerking
results_df = pd.DataFrame(results)

# Sorteren op starttijd voor overzicht
results_df = results_df.sort_values(by='Start Time')

# Print of exporteer het resultaat
print(results_df)







