import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def perfect_hindsight_strategy(df_grouped, initial_value=1000, plot=True):
    '''
    Perfect Hindsight Strategy

    Parameters
    ----------
    df_grouped : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - cik: CIK

    initial_value : float
        Initial value to start the algorithm

    Returns
    -------
    perfect_hindsight : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - cik: CIK
            - weighted_performance: Weighted Performance
            - weighted_risk: Weighted Risk
            - value: Value
    '''
    # Select the best performing cik each new_fdate
    # Ensure the 'new_fdate' is a datetime type
    df_grouped['new_fdate'] = pd.to_datetime(df_grouped['new_fdate'])
    
    # Select the best performing cik each new_fdate
    perfect_hindsight = df_grouped.loc[df_grouped.groupby('new_fdate')['weighted_performance'].idxmax()]
    perfect_hindsight.sort_values(by='new_fdate', inplace=True)

    # Initialize value for perfect hindsight strategy
    perfect_hindsight['value'] = initial_value
    for i in range(1, len(perfect_hindsight)):
        perfect_hindsight.loc[perfect_hindsight.index[i], 'value'] = perfect_hindsight.loc[perfect_hindsight.index[i-1], 'value'] * (1 + perfect_hindsight.loc[perfect_hindsight.index[i], 'weighted_performance'])

    if plot:
        # Create a figure and a set of subplots for cumulative performance
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)

        # Plot 'Value'
        axes[0].plot(perfect_hindsight['new_fdate'], perfect_hindsight['value'])
        axes[0].set_title('Value')
        axes[0].legend(['Value'])
        axes[0].grid(True)

        # Plot 'Weighted Performance'
        axes[1].plot(perfect_hindsight['new_fdate'], perfect_hindsight['weighted_performance'])
        axes[1].set_title('Weighted Performance')
        axes[1].legend(['Weighted Performance'])
        axes[1].grid(True)

        # Plot 'Weighted Risk'
        axes[2].plot(perfect_hindsight['new_fdate'], perfect_hindsight['weighted_risk'])
        axes[2].set_title('Weighted Risk')
        axes[2].legend(['Weighted Risk'])
        axes[2].grid(True)

        # Plot 'Risk Reward Ratio'
        axes[3].plot(perfect_hindsight['new_fdate'], perfect_hindsight['risk_reward_ratio'])
        axes[3].set_title('Risk Reward Ratio')
        axes[3].legend(['Risk Reward Ratio'])
        axes[3].grid(True)

        # Adjust layout for cumulative performance plots
        plt.tight_layout()
        plt.show()

    # Get the top performer for each quarter
    df_grouped['quarter'] = df_grouped['new_fdate'].dt.to_period('Q')
    top_performers = df_grouped.loc[df_grouped.groupby('quarter')['weighted_performance'].idxmax()]

    # Extract the returns and CIKs of the top performers
    returns = top_performers['weighted_performance']
    cik_labels = top_performers['cik'].astype(str) + ' (' + top_performers['quarter'].astype(str) + ')'

    if plot:
        # Plot the returns in a bar chart
        plt.figure(figsize=(15, 8))
        bars = plt.bar(cik_labels, returns, edgecolor='black', alpha=0.7)
        plt.title('Returns of Top Performers Each Quarter')
        plt.xlabel('CIK (Quarter)')
        plt.ylabel('Quarterly Return')
        plt.xticks(rotation=90)
        plt.grid(True, axis='y')

        # Annotate the bars with the return values, rotated to avoid overlap
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', rotation=90)

        plt.tight_layout()
        plt.show()

    return perfect_hindsight, top_performers

def equal_weighted_strategy(df_grouped, initial_value=1000, plot=True):
    '''
    Equal-Weighted Strategy
    
    Parameters
    ----------
    df_grouped : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - cik: CIK
            - weighted_performance: Weighted Performance
            - weighted_risk: Weighted Risk
    initial_value : float
        Initial value to start the algorithm
    
    Returns
    -------
    equal_weighted : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - Weighted Performance: Weighted Performance
            - Weighted Risk: Weighted Risk
            - Risk Reward Ratio: Risk Reward Ratio
            - Value: Value
    '''
    # Get unique new_fdates
    unique_dates = df_grouped['new_fdate'].unique()
    
    # Initialize a DataFrame to store the equal-weighted portfolio metrics
    equal_weighted = pd.DataFrame(index=unique_dates)
    
    # Calculate equal weights for each date
    for date in unique_dates:
        subset = df_grouped[df_grouped['new_fdate'] == date]
        equal_weighted.loc[date, 'Weighted Performance'] = subset['weighted_performance'].mean()
        equal_weighted.loc[date, 'Weighted Risk'] = subset['weighted_risk'].mean()
    
    # Calculate Risk Reward Ratio
    equal_weighted['Risk Reward Ratio'] = equal_weighted['Weighted Performance'] / equal_weighted['Weighted Risk']
    
    # Initialize value for equal-weighted strategy
    equal_weighted['Value'] = initial_value
    for i in range(1, len(equal_weighted)):
        equal_weighted.iloc[i, equal_weighted.columns.get_loc('Value')] = (
            equal_weighted.iloc[i - 1, equal_weighted.columns.get_loc('Value')] * 
            (1 + equal_weighted.iloc[i, equal_weighted.columns.get_loc('Weighted Performance')])
        )
    
    if plot:
        # Create a figure and a set of subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)

        # Plot 'Value'
        axes[0].plot(equal_weighted.index, equal_weighted['Value'])
        axes[0].set_title('Value')
        axes[0].legend(['Value'])
        axes[0].grid(True)

        # Plot 'Weighted Performance'
        axes[1].plot(equal_weighted.index, equal_weighted['Weighted Performance'])
        axes[1].set_title('Weighted Performance')
        axes[1].legend(['Weighted Performance'])
        axes[1].grid(True)

        # Plot 'Weighted Risk'
        axes[2].plot(equal_weighted.index, equal_weighted['Weighted Risk'])
        axes[2].set_title('Weighted Risk')
        axes[2].legend(['Weighted Risk'])
        axes[2].grid(True)

        # Plot 'Risk Reward Ratio'
        axes[3].plot(equal_weighted.index, equal_weighted['Risk Reward Ratio'])
        axes[3].set_title('Risk Reward Ratio')
        axes[3].legend(['Risk Reward Ratio'])
        axes[3].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    return equal_weighted

def greedy_strategy(df_grouped, beta_values, initial_value=1000):
    '''
    Greedy Strategy
    
    Parameters
    ----------
    df_grouped : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - cik: CIK
            - weighted_performance: Weighted Performance
            - weighted_risk: Weighted Risk
    beta_values : list
        A list of beta values to perform grid search
    initial_value : float
        Initial value to start the algorithm
    
    Returns
    -------
    best_performance : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - Performance: Performance
            - Risk: Risk
            - Value: Value
            - Risk Reward Ratio: Risk Reward Ratio
    best_weights : DataFrame
        A DataFrame with the best weights for each CIK at each date
    '''
    def calculate_performance(df_grouped, beta):
        df_grouped['new_fdate'] = pd.to_datetime(df_grouped['new_fdate'])
        df_grouped['quarter'] = df_grouped['new_fdate'].dt.to_period('Q')
        
        performance_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_performance')
        risk_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_risk')

        # Initialize weights
        weights = pd.DataFrame(1, index=performance_pivot.index, columns=performance_pivot.columns)

        # Initialize DataFrame to store cumulative performance
        cumulative_performance = pd.DataFrame(index=performance_pivot.index, columns=['Performance', 'Risk', 'Value'])
        cumulative_performance['Value'] = initial_value

        for i in range(1, len(weights)):
            date = weights.index[i]
            previous_date = weights.index[i-1]
            
            # Find the top performer
            top_performer = performance_pivot.loc[previous_date].idxmax()
            
            # Reduce weights for all except the top performer
            for cik in weights.columns:
                if cik != top_performer:
                    weights.at[date, cik] = weights.at[previous_date, cik] * beta
                else:
                    weights.at[date, cik] = weights.at[previous_date, cik]
                    
            # Normalize weights
            weights.loc[date] /= weights.loc[date].sum()

            # Calculate performance and risk
            cumulative_performance.at[date, 'Performance'] = (weights.loc[previous_date] * performance_pivot.loc[previous_date]).sum()
            cumulative_performance.at[date, 'Risk'] = (weights.loc[previous_date] * risk_pivot.loc[previous_date]).sum()
            cumulative_performance.at[date, 'Value'] = cumulative_performance.at[previous_date, 'Value'] * (1 + cumulative_performance.at[date, 'Performance'])

        cumulative_performance['Risk Reward Ratio'] = cumulative_performance['Performance'] / cumulative_performance['Risk']
        
        return cumulative_performance, weights

    # Optimize for the best beta
    best_beta = None
    best_final_value = -np.inf
    best_performance = None
    best_weights = None

    for beta in beta_values:
        performance, weights = calculate_performance(df_grouped, beta)
        final_value = performance['Value'].iloc[-1]

        if final_value > best_final_value:
            best_final_value = final_value
            best_beta = beta
            best_performance = performance
            best_weights = weights

    print(f"Optimal beta: {best_beta}")
    print(f"Final value with optimal beta: {best_final_value}")

    # Plot cumulative performance metrics
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)

    # Plot 'Value'
    axes[0].plot(best_performance.index, best_performance['Value'])
    axes[0].set_title('Value')
    axes[0].legend(['Value'])
    axes[0].grid(True)

    # Plot 'Performance'
    axes[1].plot(best_performance.index, best_performance['Performance'])
    axes[1].set_title('Performance')
    axes[1].legend(['Performance'])
    axes[1].grid(True)

    # Plot 'Risk'
    axes[2].plot(best_performance.index, best_performance['Risk'])
    axes[2].set_title('Risk')
    axes[2].legend(['Risk'])
    axes[2].grid(True)

    # Plot 'Risk Reward Ratio'
    axes[3].plot(best_performance.index, best_performance['Risk Reward Ratio'])
    axes[3].set_title('Risk Reward Ratio')
    axes[3].legend(['Risk Reward Ratio'])
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot final weights by percentage for each CIK
    final_weights = best_weights.iloc[-1] * 100
    final_weights = final_weights.sort_values(ascending=False)
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(final_weights.index, final_weights.values, edgecolor='black', alpha=0.7)
    plt.title('Final Weights by Percentage for Each CIK')
    plt.xlabel('CIK')
    plt.ylabel('Final Weight (%)')
    plt.xticks(rotation=90)
    plt.grid(True, axis='y')

    # Annotate the bars with the weight values
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', rotation=90)

    plt.tight_layout()
    plt.show()

    return best_performance, best_weights

def weighted_majority_algorithm(df_grouped, beta_values, initial_value=1000, plot=True):
    '''
    
    Weighted Majority Algorithm
    
    Parameters
    ----------
    df_grouped : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - cik: CIK
            - weighted_performance: Weighted Performance
            - weighted_risk: Weighted Risk
    beta_values : list
        A list of beta values to perform grid search
    initial_value : float
        Initial value to start the algorithm
    '''
    def calculate_wma(performance_pivot, beta):
        WMA = pd.DataFrame(np.nan, index=performance_pivot.index, columns=performance_pivot.columns)
        # Initialize weights for the first date
        for cik in WMA.columns:
            WMA[cik][0] = 1
        
        # Update weights based on performance
        for cik in WMA.columns:
            for j in range(1, len(WMA)):
                if performance_pivot[cik][j-1] > 0:
                    WMA[cik][j] = WMA[cik][j-1]
                else:
                    WMA[cik][j] = WMA[cik][j-1] * beta
        
        # Normalize weights
        WMA['WMA'] = WMA.sum(axis=1)
        for row in range(len(WMA)):
            WMA.iloc[row, :-1] = WMA.iloc[row, :-1] / WMA.iloc[row, -1]
        
        # Calculate Performance, Risk, Risk Reward Ratio, and Value
        WMA['Performance'] = WMA.iloc[:, :-2].mul(performance_pivot).sum(axis=1)
        WMA['Risk'] = WMA.iloc[:, :-2].mul(df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_risk')).sum(axis=1)
        WMA['Risk Reward Ratio'] = WMA['Performance'] / WMA['Risk']
        WMA['Value'] = initial_value
        
        for i in range(1, len(WMA)):
            WMA.loc[WMA.index[i], 'Value'] = WMA.loc[WMA.index[i-1], 'Value'] * (1 + WMA.loc[WMA.index[i], 'Performance'])
        
        return WMA

    # Perform grid search over beta values
    performance_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_performance')
    final_values = []

    for beta in beta_values:
        WMA = calculate_wma(performance_pivot, beta)
        final_values.append(WMA['Value'].iloc[-1])

    # Find the optimal beta value
    optimal_beta = beta_values[np.argmax(final_values)]
    optimal_final_value = max(final_values)

    print(f"Optimal beta: {optimal_beta}")
    print(f"Final value with optimal beta: {optimal_final_value}")

    # Plot the final values for different beta values
    plt.figure(figsize=(10, 5))
    plt.plot(beta_values, final_values)
    plt.xlabel('Beta')
    plt.ylabel('Final Value')
    plt.title('Final Value for Different Beta Values')
    plt.grid(True)
    plt.show()

    # Run the WMA algorithm with the optimal beta
    WMA = calculate_wma(performance_pivot, optimal_beta)

    if plot:
        # Create a figure and a set of subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)

        # Plot 'Value'
        axes[0].plot(WMA.index, WMA['Value'])
        axes[0].set_title('Value')
        axes[0].legend(['Value'])
        axes[0].grid(True)

        # Plot 'Performance'
        axes[1].plot(WMA.index, WMA['Performance'])
        axes[1].set_title('Performance')
        axes[1].legend(['Performance'])
        axes[1].grid(True)

        # Plot 'Risk'
        axes[2].plot(WMA.index, WMA['Risk'])
        axes[2].set_title('Risk')
        axes[2].legend(['Risk'])
        axes[2].grid(True)

        # Plot 'Risk Reward Ratio'
        axes[3].plot(WMA.index, WMA['Risk Reward Ratio'])
        axes[3].set_title('Risk Reward Ratio')
        axes[3].legend(['Risk Reward Ratio'])
        axes[3].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    return WMA, optimal_beta, optimal_final_value

def randomized_weighted_majority_algorithm(df_grouped, initial_value=1000, beta=1.1, plot = True):
    '''
    Randomized Weighted Majority Algorithm
    
    Parameters
    ----------
    df_grouped : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - cik: CIK
            - weighted_performance: Weighted Performance
            - weighted_risk: Weighted Risk
    beta_values : list
        A value of beta to perform the algorithm
    initial_value : float
        Initial value to start the algorithm

    Returns
    -------
    metrics : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - Performance: Performance
            - Risk: Risk
            - Risk Reward Ratio: Risk Reward Ratio
            - Value: Value
    '''
    def calculate_rwma(performance_pivot):
        # Initialize all experts' weights to 1
        WMA = pd.DataFrame(1, index=performance_pivot.index, columns=performance_pivot.columns)

        # Normalize weights to probabilities for the first round
        WMA = WMA.div(WMA.sum(axis=1), axis=0)

        # Initialize performance metrics
        metrics = pd.DataFrame(index=performance_pivot.index, columns=['Performance', 'Risk', 'Risk Reward Ratio', 'Value'])
        metrics['Value'] = initial_value
        
        # Iterate through each round (date)
        for i in range(1, len(WMA)):
            current_date = WMA.index[i]
            previous_date = WMA.index[i-1]
            
            # Total weight
            total_weight = WMA.loc[previous_date].sum()

            # Choose expert i randomly with probability w_i / W
            probabilities = WMA.loc[previous_date] / total_weight
            selected_expert = np.random.choice(WMA.columns, p=probabilities)

            # Predict as the chosen expert predicts (update performance and risk)
            metrics.at[current_date, 'Performance'] = performance_pivot.at[previous_date, selected_expert]
            metrics.at[current_date, 'Risk'] = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_risk').at[previous_date, selected_expert]
            metrics.at[current_date, 'Value'] = metrics.at[previous_date, 'Value'] * (1 + metrics.at[current_date, 'Performance'])

            # Multiply the weights of all experts who predicted wrongly by beta
            for cik in WMA.columns:
                if performance_pivot.at[previous_date, cik] != performance_pivot.at[previous_date, selected_expert]:
                    WMA.at[current_date, cik] = WMA.at[previous_date, cik] * beta
                else:
                    WMA.at[current_date, cik] = WMA.at[previous_date, cik]

            # Normalize weights for the next round
            total_weight = WMA.loc[current_date].sum()
            WMA.loc[current_date] = WMA.loc[current_date] / total_weight

        metrics['Risk Reward Ratio'] = metrics['Performance'] / metrics['Risk']
        return metrics

    # Pivot the performance data
    performance_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_performance')

    # Calculate RWMA with fixed beta
    metrics = calculate_rwma(performance_pivot)

    if plot:
        # Create a figure and a set of subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)

        # Plot 'Value'
        axes[0].plot(metrics.index, metrics['Value'])
        axes[0].set_title('Value')
        axes[0].legend(['Value'])
        axes[0].grid(True)

        # Plot 'Performance'
        axes[1].plot(metrics.index, metrics['Performance'])
        axes[1].set_title('Performance')
        axes[1].legend(['Performance'])
        axes[1].grid(True)

        # Plot 'Risk'
        axes[2].plot(metrics.index, metrics['Risk'])
        axes[2].set_title('Risk')
        axes[2].legend(['Risk'])
        axes[2].grid(True)

        # Plot 'Risk Reward Ratio'
        axes[3].plot(metrics.index, metrics['Risk Reward Ratio'])
        axes[3].set_title('Risk Reward Ratio')
        axes[3].legend(['Risk Reward Ratio'])
        axes[3].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    return metrics
