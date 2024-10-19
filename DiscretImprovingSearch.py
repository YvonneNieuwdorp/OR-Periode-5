import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

# Laad de gegevens
Orders = pd.read_excel('PaintShop - November 2024.xlsx', sheet_name='Orders')
Machines = pd.read_excel('PaintShop - November 2024.xlsx', sheet_name='Machines')
Setups = pd.read_excel('PaintShop - November 2024.xlsx', sheet_name='Setups')
schedule_best_costs = pd.read_excel("Schedule Constructive Heuristic Best Costs.xlsx")
schedule_deadline = pd.read_excel("Schedule Constructive Heuristic Deadline.xlsx")


def processing_time(ordernumber, number_of_machine):
    """Berekent processing time
    Parameters: ordernumber (int) en number_of_machine (int)
    output: processing time (int)
    """
    return Orders.loc[ordernumber, 'Surface'] / Machines.loc[number_of_machine, 'Speed']

def setup_time(prev_colour, new_colour):
    """Berekent setup time
    Parameters: previous colour (str) en new_colour (str)
    output: setup time (int)
    """
    result = Setups.loc[(Setups['From colour'] == prev_colour) & (Setups['To colour'] == new_colour), 'Setup time']
    if not result.empty:
        return result.values[0]
    else:
        return 0 

def total_schedule_cost(new_schedule):
    """Berekent de totale kosten van de nieuwe planning en update het schema.
    
    Parameters:
        new_schedule: DataFrame met de nieuwe scheduling resultaten.
        
    Output: 
        Totaal van penaltykosten.
    """
    total_penalty_cost = 0
    current_time = {f'M{i+1}': 0 for i in range(4)}  # Voor 4 machines M1, M2, M3, M4
    current_colours = {f'M{i+1}': None for i in range(4)}  # Voor kleuren

    # Itereer door elke order in de nieuwe planning
    for idx, row in new_schedule.iterrows():
        order_idx = Orders[Orders['Order'] == row['Order']].index[0]  # Vind het order index
        machine_idx = {"M1": 0, "M2": 1, "M3": 2, "M4": 3}[row["Machine"]]
        
        # Setup en processing tijd berekenen
        setup = setup_time(current_colours[row["Machine"]], row["Colour"])
        processing = processing_time(order_idx, machine_idx)
        
        # Bereken start- en eindtijd
        start_time = current_time[row["Machine"]] + setup
        finish_time = start_time + processing
        
        # Update huidige tijd en kleur voor de machine
        current_time[row["Machine"]] = finish_time
        current_colours[row["Machine"]] = row["Colour"]
        
        # Bereken lateness en penalty kosten
        lateness = max(0, finish_time - Orders.loc[order_idx, 'Deadline'])
        penalty_cost = lateness * Orders.loc[order_idx, 'Penalty']
        
        # Update de DataFrame met de nieuwe waarden
        new_schedule.at[idx, 'Start Time'] = start_time
        new_schedule.at[idx, 'Finish Time'] = finish_time
        new_schedule.at[idx, 'Setup Time'] = setup
        new_schedule.at[idx, 'Processing Time'] = processing
        new_schedule.at[idx, 'Lateness'] = lateness
        new_schedule.at[idx, 'Penalty Cost'] = penalty_cost
        
        # Voeg de penalty kosten toe aan de totale kosten
        total_penalty_cost += penalty_cost

    return total_penalty_cost

def two_opt(schedule_df):
    """Voert Discrete Improving Search uit
    Parameters: schedule_df (DataFrame)
    Output: 
        - schedule_df: DataFrame, improved schedule
        - costs_history: list, costs per DataFrame per iterations
    """
    improved = True
    costs_history = []  # Lijst om de kosten van elke iteratie op te slaan
    
    while improved:
        improved = False
        # Loop over alle mogelijke combinaties van orders
        for i in range(len(schedule_df) - 1):
            for j in range(i + 1, len(schedule_df)):
                
                new_schedule_df = schedule_df.copy()
                
                # Swap de orders op posities i en j
                new_schedule_df.iloc[i:j + 1] = new_schedule_df.iloc[i:j + 1][::-1]

                # Bereken de totale kosten van het nieuwe schema
                new_cost = total_schedule_cost(new_schedule_df)
                current_cost = total_schedule_cost(schedule_df)

                # Vergelijk kosten en update als er een verbetering is
                if new_cost < current_cost:
                    schedule_df = new_schedule_df
                    improved = True
        
        # Voeg de kosten van deze iteratie toe aan de history
        costs_history.append(total_schedule_cost(schedule_df))

    return schedule_df, costs_history  # Return ook de kosten geschiedenis

def plot_gantt_chart(schedule_df):
    """Visualize the scheduling process using a Gantt chart, showing both setup (grey) and processing times (colored).
    
    Parameters:
        schedule_df: DataFrame with columns 'Machine', 'Order', 'Start Time', 'Finish Time', 'Colour', 'Setup Time', and 'Processing Time'.
    Output: Gantt chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colours = {
        "Red": "red",
        "Blue": "blue",
        "Green": "green",
        "Yellow": "yellow",
        "Purple": "purple",
        "Setup": "grey"
    }

    for idx, row in schedule_df.iterrows():
        machine = row["Machine"]
        start_time = row["Start Time"]
        setup_duration = row["Setup Time"]
        processing_duration = row["Processing Time"]
        order_colour = row["Colour"]
        order_name = row["Order"]

        # Index voor de machine-positie
        machine_idx = {"M1": 0, "M2": 1, "M3": 2, "M4": 3}[machine]

        # Setup time bar (grijs), vóór de verwerkingstijd
        setup_start = start_time - setup_duration  # Setup begint op de oorspronkelijke starttijd
        ax.barh(machine_idx, setup_duration, left=setup_start, color=colours["Setup"], edgecolor='black', label="Setup" if idx == 0 else "")
        
        # Processing time bar (gekleurde balk voor de order), ná de setup-tijd
        processing_start = start_time  # Verplaats de starttijd van de order na de setup-tijd
        ax.barh(machine_idx, processing_duration, left=processing_start, color=colours[order_colour], edgecolor='black')
        
        # Label in het midden van de verwerkingstijd
        ax.text(processing_start + (processing_duration / 2), machine_idx, order_name, va='center', ha='center', color='white', fontsize=10)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["M1", "M2", "M3", "M4"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title("Gantt Chart of Order Scheduling with Setup and Processing Times")

    # Legend for order colors and setup time
    legend_patches = [mpatches.Patch(color=col, label=label) for label, col in colours.items()]
    ax.legend(handles=legend_patches, title="Order Colours and Setup Time")

    plt.show()

start_time = time.time()  # Start tijd opnemen
improved_schedule_2opt, costs_history = two_opt(schedule_deadline)  # Hier neem je je initiële planning mee
end_time = time.time()  # Eindtijd opnemen

# Resultaten tonen
print(f"Initial cost: {total_schedule_cost(schedule_deadline):.2f}")
print(f"Improved cost: {total_schedule_cost(improved_schedule_2opt):.2f}")
print(f"Time taken: {end_time - start_time:.2f} seconds")  # Tijd berekenen en printen

# Plot de Gantt diagrammen
plot_gantt_chart(schedule_deadline)
plot_gantt_chart(improved_schedule_2opt)

# Plot de kosten tegenover de iteraties
plt.figure(figsize=(10, 5))
plt.plot(costs_history, marker='o')
plt.title("Kosten tegenover Iteraties tijdens 2-opt Optimalisatie")
plt.xlabel("Iteratie")
plt.ylabel("Totale Kosten")
plt.grid()
plt.show()
