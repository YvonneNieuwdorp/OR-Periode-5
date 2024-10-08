import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
import time

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

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Start Time')


def total_schedule_cost(results_df):
    """ Berekent de totale kosten van de schedule
    Parameters:
        - results_df: DataFrame met de resultaten van het scheduling proces
    Output: 
        - Totaal van penaltykosten 
    """
    return results_df['Penalty Cost'].sum()


def swap_orders(results_df):
    """Swap two randomly selected orders in the schedule.
    Parameters: 
     - results_df: DataFrame, complete schedule
    Output: 
     - new_schedule: DataFrame, new schedule, different to original
    """
    new_schedule = results_df.copy()
    
    idx1, idx2 = random.sample(range(len(new_schedule)), 2) # Randomly select two different indices to swap 
    temp = new_schedule.iloc[idx1].copy() # Swap the two orders in the dataframe
    new_schedule.iloc[idx1] = new_schedule.iloc[idx2]
    new_schedule.iloc[idx2] = temp
    
    return new_schedule

def reassign_machine(results_df):
    """Reassign a randomly selected order to a different machine.
    Parameters: 
     - results_df: DataFrame, complete schedule
    Output: 
     - new_schedule: DataFrame, new schedule, different to original
     """
    new_schedule = results_df.copy()
    
    idx = random.randint(0, len(new_schedule) - 1) # Randomly select an index to change the machine
       
    machines = ["M1", "M2", "M3"]  # List of machines
    current_machine = new_schedule.iloc[idx]["Machine"]
    new_machine = random.choice([m for m in machines if m != current_machine]) # Select a new machine different from the current one
    new_schedule.at[idx, "Machine"] = new_machine # Update the machine assignment in the dataframe
    
    return new_schedule

def discrete_improving_search(results_df, iterations):
    """Perform Discrete Improving Search to improve the schedule.
    Parameters
    - results_df: DataFrame, complete working DataFrame
    - iterations: Int, number of iterations
    Output: 
    - best_schedule: DataFrame, schedule with a local optimum
    """
    best_schedule = results_df.copy()
    best_cost = total_schedule_cost(best_schedule)
    
    for i in range(iterations):
        # Generate a new schedule by either swapping orders or reassigning a machine
        if random.random() < 0.5:
            new_schedule = swap_orders(best_schedule)
        else:
            new_schedule = reassign_machine(best_schedule)
        
        # Recalculate the start times, finish times, lateness, and penalty costs
        for idx, row in new_schedule.iterrows():
            order_idx = Orders[Orders['Order'] == row['Order']].index[0]
            machine_idx = {"M1": 0, "M2": 1, "M3": 2}[row["Machine"]]
            setup = setup_time(new_schedule.iloc[idx-1]["Colour"] if idx > 0 else None, row["Colour"])
            processing = processing_time(order_idx, machine_idx)
            start_time = new_schedule.iloc[idx-1]["Finish Time"] if idx > 0 else 0 + setup
            finish_time = start_time + processing
            lateness = max(0, finish_time - Orders.loc[order_idx, 'Deadline'])
            penalty_cost = lateness * Orders.loc[order_idx, 'Penalty']
            
            # Update the dataframe with the new values
            new_schedule.at[idx, 'Start Time'] = start_time
            new_schedule.at[idx, 'Finish Time'] = finish_time
            new_schedule.at[idx, 'Setup Time'] = setup
            new_schedule.at[idx, 'Processing Time'] = processing
            new_schedule.at[idx, 'Lateness'] = lateness
            new_schedule.at[idx, 'Penalty Cost'] = penalty_cost
        
        # Calculate the total cost of the new schedule
        new_cost = total_schedule_cost(new_schedule)
        
        # If the new schedule is better, update the best schedule
        if new_cost < best_cost:
            best_schedule = new_schedule.copy()
            best_cost = new_cost
    
    return best_schedule

start_time = time.time()  # Start tijd opnemen
improved_schedule = discrete_improving_search(results_df, iterations=10000)
end_time = time.time()  # Eindtijd opnemen

print(f"Initial cost: {total_schedule_cost(results_df):.2f}")
print(f"Improved cost: {total_schedule_cost(improved_schedule):.2f}")
print(f"Time taken: {end_time - start_time:.2f} seconds")  # Tijd berekenen en printen

def plot_gantt_chart(schedule_df):
    """Visualize the scheduling process using a Gantt chart with setup and processing times indicated separately.
    
    Parameters:
        schedule_df: DataFrame with columns 'Machine', 'Order', 'Start Time', 'Finish Time', 'Colour', 'Setup Time', and 'Processing Time'.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colours = {
        "Red": "red",
        "Blue": "blue",
        "Green": "green",
        "Yellow": "yellow",
        "Setup": "grey"  # Color for setup time
    }

    for idx, row in schedule_df.iterrows():
        machine = row["Machine"]
        start = row["Start Time"]
        setup_duration = row["Setup Time"]
        processing_duration = row["Processing Time"]
        order_colour = row["Colour"]
        order_name = row["Order"]

        # Index voor de machine-positie
        machine_idx = {"M1": 0, "M2": 1, "M3": 2}[machine]

        # Setup time bar
        ax.barh(machine_idx, setup_duration, left=start, color=colours["Setup"], edgecolor='black')

        # Processing time bar
        ax.barh(machine_idx, processing_duration, left=start + setup_duration, color=colours[order_colour], edgecolor='black')
        
        # Label in the center of the processing time
        ax.text(start + setup_duration + (processing_duration / 2), machine_idx, order_name, va='center', ha='center', color='white', fontsize=10)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["M1", "M2", "M3"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title("Gantt Chart of Order Scheduling with Setup and Processing Times")

    # Legend for order colors and setup time
    legend_patches = [mpatches.Patch(color=col, label=label) for label, col in colours.items()]
    ax.legend(handles=legend_patches, title="Order Colours and Setup Time")

    plt.show()

# Voorbeeld aanroepen van de functie met de results_df DataFrame
plot_gantt_chart(results_df)

# Sla de verbeterde planning op in een Excel-bestand
output_file = 'improved_schedule.xlsx'
#improved_schedule.to_excel(output_file, index=False)

plot_gantt_chart(improved_schedule)

