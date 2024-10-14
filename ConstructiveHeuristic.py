import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

# Laad de gegevens
Orders = pd.read_excel('PaintShop - November 2024.xlsx', sheet_name='Orders')
Machines = pd.read_excel('PaintShop - November 2024.xlsx', sheet_name='Machines')
Setups = pd.read_excel('PaintShop - November 2024.xlsx', sheet_name='Setups')

# Functie om processing time te berekenen
def processing_time(ordernumber, number_of_machine):
    """ Functie die de processing time berekent
    Parameters: 
        - ordernummer, als een int
        - machine die gebruikt wordt, als een int
    Output: de tijd die nodig is om de order uit te voeren
    """
    return Orders.loc[ordernumber, 'Surface'] / Machines.loc[number_of_machine, 'Speed']

# Functie om de setup time te berekenen
def setup_time(prev_colour, new_colour):
    """ Functie die de setuptime berekent
    Parameters:
        - previous color als een string
        - new color als een string
    Output: setup_time
    """
    result = Setups.loc[(Setups['From colour'] == prev_colour) & (Setups['To colour'] == new_colour), 'Setup time']
    if not result.empty:
        return result.values[0]
    else:
        return 0 

## EERSTE SCHEMA: Best Costs
def constructive_heuristic_best_costs(current_times, available_orders, current_colours):
    best_order = None
    best_machine = None
    min_cost = float('inf')

    for index, order in available_orders.iterrows(): 
        for i in range(4):  # Loop over 4 machines
            time_to_finish = current_times[i] + processing_time(index, i)
            lateness = max(0, time_to_finish - order['Deadline'])
            cost = lateness * order['Penalty']
            cost += setup_time(current_colours[i], order["Colour"]) if current_colours[i] else 0

            if cost < min_cost:
                min_cost = cost
                best_order = index
                best_machine = f"M{i+1}"

    return best_order, best_machine

# Lijst om iteratieresultaten op te slaan voor constructive heuristic best costs
results_ch = []

# Herinitialiseren van tijden en kleuren voor de eerste planningsronde
current_times = [0] * 4
available_orders_ch = Orders.copy()
current_colours = [None] * 4

# Scheduling loop voor nearest neighbor
while not available_orders_ch.empty:
    next_order_idx, best_machine = constructive_heuristic_best_costs(current_times, available_orders_ch, current_colours)
    
    if next_order_idx is not None:
        next_order = available_orders_ch.loc[next_order_idx]
        machine_index = int(best_machine[1]) - 1  # Omzetten van machine string naar index
        
        # Bereken de starttijd, verwerkingstijd en setuptijd
        setup = setup_time(current_colours[machine_index], next_order["Colour"])
        processing = processing_time(next_order_idx, machine_index)
        start_time = current_times[machine_index] + setup
        finish_time = start_time + processing
        lateness = max(0, finish_time - next_order['Deadline'])
        current_times[machine_index] = finish_time
        current_colours[machine_index] = next_order["Colour"]

        # Voeg de huidige status van de planning toe aan de resultaten
        results_ch.append({
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
        available_orders_ch.drop(next_order_idx, inplace=True)

# Zet resultaten om in een DataFrame voor nearest neighbor
results_df_ch = pd.DataFrame(results_ch)
results_df_ch = results_df_ch.sort_values(by='Start Time')

## TWEEDE SCHEMA: DEADLINE
def constructive_heuristic_deadline(current_times, available_orders, current_colours):
    best_order = None
    best_machine = None
    min_time_to_deadline = float('inf')  # Begin met een hoge waarde

    for index, order in available_orders.iterrows():
        for i in range(4):  # Loop over 4 machines
            time_to_finish = current_times[i] + setup_time(current_colours[i], order["Colour"]) + processing_time(index, i)
            time_to_deadline = max(0, order['Deadline'] - time_to_finish)

            # Kies het order dat het dichtst bij de deadline ligt en controleer de machine met de minste werkdruk
            if time_to_deadline < min_time_to_deadline and current_times[i] <= min(current_times[:i] + current_times[i+1:]):
                min_time_to_deadline = time_to_deadline
                best_order = index
                best_machine = f"M{i+1}"

    return best_order, best_machine

# Lijst om iteratieresultaten op te slaan voor deadline
results_dl = []

# Herinitialiseren van tijden en kleuren voor de tweede planningsronde
current_times = [0] * 4
available_orders_dl = Orders.copy()
current_colours = [None] * 4

# Scheduling loop voor deadline
while not available_orders_dl.empty:
    next_order_idx, best_machine = constructive_heuristic_deadline(current_times, available_orders_dl, current_colours)
    
    if next_order_idx is not None:
        next_order = available_orders_dl.loc[next_order_idx]
        machine_index = int(best_machine[1]) - 1  # Omzetten van machine string naar index
        
        # Bereken de starttijd, verwerkingstijd en setuptijd
        setup = setup_time(current_colours[machine_index], next_order["Colour"])
        processing = processing_time(next_order_idx, machine_index)
        start_time = current_times[machine_index] + setup
        finish_time = start_time + processing
        lateness = max(0, finish_time - next_order['Deadline'])
        current_times[machine_index] = finish_time
        current_colours[machine_index] = next_order["Colour"]

        # Voeg de huidige status van de planning toe aan de resultaten
        results_dl.append({
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
        available_orders_dl.drop(next_order_idx, inplace=True)

# Zet resultaten om in een DataFrame voor deadline
results_df_dl = pd.DataFrame(results_dl)
results_df_dl = results_df_dl.sort_values(by='Start Time')

# Print de resultaten
print("Results from Nearest Neighbor Scheduling:")
print(results_df_ch)

print("\nResults from Deadline Scheduling:")
print(results_df_dl)
