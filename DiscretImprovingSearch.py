import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
import time

schedule_best_costs = pd.read_excel('Schedule Constructive Heuristic Best Costs')
schedule_deadline = pd.read_excel('Schedule Constructive Heuristic Deadline')

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
    costs = []  # Lijst om kosten per iteratie op te slaan
    
    for i in range(iterations):
        # Generate a new schedule by either swapping orders or reassigning a machine
        if random.random() < 0.5:
            new_schedule = swap_orders(best_schedule)
        else:
            new_schedule = reassign_machine(best_schedule)
        
        # Recalculate the start times, finish times, lateness, and penalty costs
        new_schedule = new_schedule.copy()  # Create a copy for the new schedule
        current_time = {'M1': 0, 'M2': 0, 'M3': 0}  # Initialize current times for all machines
        current_colours = {'M1': None, 'M2': None, 'M3': None}  # Initialize current colours for all machines
        
        for idx, row in new_schedule.iterrows():
            order_idx = Orders[Orders['Order'] == row['Order']].index[0]
            machine_idx = {"M1": 0, "M2": 1, "M3": 2}[row["Machine"]]
            setup = setup_time(current_colours[row["Machine"]], row["Colour"])
            processing = processing_time(order_idx, machine_idx)
            start_time = current_time[row["Machine"]] + setup
            finish_time = start_time + processing
            
            # Update current time and colour for the machine
            current_time[row["Machine"]] = finish_time
            current_colours[row["Machine"]] = row["Colour"]
            
            # Update the dataframe with the new values
            new_schedule.at[idx, 'Start Time'] = start_time
            new_schedule.at[idx, 'Finish Time'] = finish_time
            new_schedule.at[idx, 'Setup Time'] = setup
            new_schedule.at[idx, 'Processing Time'] = processing
            new_schedule.at[idx, 'Lateness'] = max(0, finish_time - Orders.loc[order_idx, 'Deadline'])
            new_schedule.at[idx, 'Penalty Cost'] = new_schedule.at[idx, 'Lateness'] * Orders.loc[order_idx, 'Penalty']
        
        # Calculate the total cost of the new schedule
        new_cost = total_schedule_cost(new_schedule)
        costs.append(new_cost)  # Voeg de kosten toe aan de lijst
        
        # If the new schedule is better, update the best schedule
        if new_cost < best_cost:
            best_schedule = new_schedule.copy()
            best_cost = new_cost
    
    # Plot de kosten per iteratie
    plt.figure(figsize=(10, 5))
    plt.plot(range(iterations), costs, marker='o')
    plt.title('Kosten over Iteraties')
    plt.xlabel('Iteraties')
    plt.ylabel('Totale Kosten')
    plt.grid()
    plt.show()

    return best_schedule

start_time = time.time()  # Start tijd opnemen
improved_schedule = discrete_improving_search(results_df_nn, iterations=10000)
end_time = time.time()  # Eindtijd opnemen

print(f"Initial cost: {total_schedule_cost(results_df_nn):.2f}")
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

# Sla de verbeterde planning op in een Excel-bestand
#output_file = 'improved_schedule.xlsx'
#improved_schedule.to_excel(output_file, index=False)
plot_gantt_chart(results_df_nn)
plot_gantt_chart(improved_schedule)
