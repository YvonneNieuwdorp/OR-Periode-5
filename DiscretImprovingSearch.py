import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def processing_time(ordernumber, number_of_machine):
    """Functie die de processing time berekent.
    
    Parameters: 
        ordernummer: als een int
        machine die gebruikt wordt: als een int
    
    Output: 
        De tijd die nodig is om de order uit te voeren.
    """
    return Orders.loc[ordernumber, 'Surface'] / Machines.loc[number_of_machine, 'Speed']

def setup_time(prev_colour, new_colour):
    """Functie die de setuptime berekent.
    
    Parameters:
        previous color: als een string
        new color: als een string
        
    Output: 
        setup_time
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


def discrete_improvement_search(initial_schedule):
    """Implements a discrete improvement search algorithm to optimize the schedule.
    
    Parameters:
        initial_schedule: DataFrame with initial scheduling information.
        
    Returns:
        best_schedule: DataFrame with the local optimum schedule.
        costs: list of total costs for each iteration
    """
    # initialiseren van huidig schema
    num_orders = len(initial_schedule)
    curr_schedule = initial_schedule.copy()  # Maak een kopie van het oorspronkelijke schema
    total_costs = total_schedule_cost(curr_schedule)  # Initieel totale kosten
    iterations = 0
    
    while True:
        # bepalen improvement en gain
        max_gain = float('-inf')  # Start met een zeer negatieve gain
        fi_move_found = False
        move = None  # Houdt de best move bij
        
        for i in range(1, num_orders - 1):  # van i = 1 t/m num_orders -2
            for j in range(i + 2, num_orders + 1):  # van j = i+2 t/m num_orders
                if (i == 1 and j == num_orders): 
                    break
                
                # Bereken de indices voor de orders
                order_i = curr_schedule['Order'][i]
                order_j = curr_schedule['Order'][j - 1] if j < num_orders else curr_schedule['Order'][0]

                # Maak een tijdelijke kopie van het schema
                neighbor_schedule = curr_schedule.copy()
                
                # Voer de move uit
                neighbor_schedule.iloc[i:j] = reversed(neighbor_schedule.iloc[i:j])
                
                # Bereken de totale kosten van de nieuwe planning
                new_cost = total_schedule_cost(neighbor_schedule)
                
                # Bereken de gain
                gain = total_costs - new_cost  # Positieve gain betekent kostenbesparing
                
                # Controleer of deze move beter is
                if gain > max_gain:
                    max_gain = gain
                    move = (i, j)
                    fi_move_found = True
        
        # stap 1: als geen move is gevonden die improving en feasible is: stop
        if not fi_move_found: 
            break
        
        # stap 2: kies verbetering move
        i, j = move
        
        # stap 3: update de tour
        curr_schedule.iloc[i:j] = reversed(curr_schedule.iloc[i:j])  # Pas de move toe
        total_costs = total_schedule_cost(curr_schedule)  # Update de totale kosten
        
    return curr_schedule

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
improved_schedule = discrete_improving_search(schedule_best_costs, iterations=100)
end_time = time.time()  # Eindtijd opnemen

print(f"Initial cost: {total_schedule_cost(schedule_best_costs):.2f}")
print(f"Improved cost: {total_schedule_cost(improved_schedule):.2f}")
print(f"Time taken: {end_time - start_time:.2f} seconds")  # Tijd berekenen en printen

plot_gantt_chart(schedule_best_costs)
plot_gantt_chart(improved_schedule)
