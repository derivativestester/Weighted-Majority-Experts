import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_performance_metrics(performance):
    '''
    Plot Performance Metrics
    
    Parameters
    ----------
    performance : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - Performance: Performance
            - Risk: Risk
            - Risk Reward Ratio: Risk Reward Ratio
            - Value: Value
    '''
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)

    # Plot 'Value'
    axes[0].plot(performance.index, performance['Value'])
    # label the final
    axes[0].annotate(f'{performance["Value"].iloc[-1]:.2f}', xy=(performance.index[-1], performance['Value'].iloc[-1]),
                     xytext=(-10, 10), textcoords="offset points", ha='right', va='bottom')
    axes[0].set_title('Value')
    axes[0].legend(['Value'])
    axes[0].grid(True)

    # Plot 'Performance'
    axes[1].plot(performance.index, performance['Performance'])
    # mean performance
    mean_performance = performance['Performance'].mean()
    axes[1].axhline(y=mean_performance, color='r', linestyle='--', label=f'Mean Performance: {mean_performance:.2f}')
    axes[1].annotate(f'{mean_performance:.2f}', xy=(performance.index[-1], mean_performance), xytext=(-10, 10),
                     textcoords="offset points", ha='right', va='bottom')
    axes[1].set_title('Performance')
    axes[1].legend(['Performance'])
    axes[1].grid(True)

    # Plot 'Risk'
    axes[2].plot(performance.index, performance['Risk'])
    # mean risk
    mean_risk = performance['Risk'].mean()
    axes[2].axhline(y=mean_risk, color='r', linestyle='--', label=f'Mean Risk: {mean_risk:.2f}')
    axes[2].annotate(f'{mean_risk:.2f}', xy=(performance.index[-1], mean_risk), xytext=(-10, 10),
                     textcoords="offset points", ha='right', va='bottom')
    axes[2].set_title('Risk')
    axes[2].legend(['Risk'])
    axes[2].grid(True)

    # Plot 'Risk Reward Ratio'
    axes[3].plot(performance.index, performance['Risk Reward Ratio'])
    # mean risk reward ratio
    mean_risk_reward_ratio = performance['Risk Reward Ratio'].mean()
    axes[3].axhline(y=mean_risk_reward_ratio, color='r', linestyle='--', label=f'Mean Risk Reward Ratio: {mean_risk_reward_ratio:.2f}')
    axes[3].annotate(f'{mean_risk_reward_ratio:.2f}', xy=(performance.index[-1], mean_risk_reward_ratio), xytext=(-10, 10),
                        textcoords="offset points", ha='right', va='bottom')
    axes[3].set_title('Risk Reward Ratio')
    axes[3].legend(['Risk Reward Ratio'])
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

def plot_final_weights(weights):
    '''
    Plot Final Weights by Percentage for Each CIK
    
    Parameters
    ----------
    weights : DataFrame
        A DataFrame with weights for each CIK at each date
    '''
    final_weights = weights.iloc[-1] * 100
    final_weights = final_weights.sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    bars = plt.bar(final_weights.index, final_weights.values, edgecolor='black', alpha=0.7)
    plt.title('Final Weights by Percentage for Each CIK')
    plt.xlabel('CIK')
    plt.ylabel('Final Weight (%)')
    plt.xticks(rotation=90)
    plt.grid(True, axis='y')

    # Annotate the bars with the weight values and makes sure text does not overlap out of table
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def best_expert_strategy(df_grouped, initial_value=1000, criterion='performance', plot=True):
    '''
    Best Expert Strategy
    
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
    criterion : str
        The criterion to use for selecting the best expert ('performance', 'risk', 'risk_reward')
    plot : bool
        Whether to plot the results

    Returns
    -------
    best_expert : DataFrame
        A DataFrame with the following columns:
            - new_fdate: Date
            - cik: CIK
            - weighted_performance: Weighted Performance
            - weighted_risk: Weighted Risk
            - value: Value
            - risk_reward_ratio: Risk Reward Ratio
    '''
    # Ensure the 'new_fdate' is a datetime type
    df_grouped['new_fdate'] = pd.to_datetime(df_grouped['new_fdate'])
    
    # Calculate cumulative performance for each cik
    performance_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_performance')
    risk_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_risk')

    
    # Initialize a DataFrame to store cumulative performance
    cumulative_performance = pd.DataFrame(index=performance_pivot.index, columns=performance_pivot.columns)
    
    # Set initial value for each fund
    cumulative_performance.iloc[0] = initial_value
    
    # Calculate cumulative performance
    for cik in performance_pivot.columns:
        for i in range(1, len(performance_pivot)):
            cumulative_performance.iloc[i][cik] = cumulative_performance.iloc[i-1][cik] * (1 + performance_pivot.iloc[i][cik])
    
    if criterion == 'performance':
        best_expert = cumulative_performance.iloc[-1].idxmax()
    elif criterion == 'risk':
        best_expert = risk_pivot.iloc[-1].idxmin()
    elif criterion == 'risk_reward':
        risk_reward_pivot = performance_pivot / risk_pivot
        best_expert = risk_reward_pivot.iloc[-1].idxmax()
    else:
        raise ValueError("Invalid criterion. Choose from 'performance', 'risk', 'risk_reward'.")

    # Prepare the result DataFrame
    result = df_grouped[df_grouped['cik'] == best_expert].copy()
    result['value'] = cumulative_performance[best_expert]
    result['risk_reward_ratio'] = performance_pivot[best_expert] / risk_pivot[best_expert]

    # Plot value, performance, risk, and risk-reward ratio for the best expert
    if plot:
        plot_performance_metrics(result)

    return result

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
        plot_performance_metrics(equal_weighted)

    return equal_weighted

def greedy_strategy(df_grouped, beta_values, initial_value=1000, criterion='performance', plot=True):
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
    criterion : str
        The criterion to use for selecting the best expert ('performance', 'risk', 'risk_reward')
    
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
    beta_values : list
        List of beta values tested
    final_values : list
        List of final metric values for each beta value
    '''
    def calculate_performance(df_grouped, beta):
        df_grouped['new_fdate'] = pd.to_datetime(df_grouped['new_fdate'])
        
        performance_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_performance')
        risk_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_risk')

        # Initialize weights with equal weights for the first date
        initial_weights = 1 / performance_pivot.shape[1]
        weights = pd.DataFrame(initial_weights, index=performance_pivot.index, columns=performance_pivot.columns)

        # Initialize DataFrame to store cumulative performance
        cumulative_performance = pd.DataFrame(index=performance_pivot.index, columns=['Performance', 'Risk', 'Value'])
        cumulative_performance['Value'] = initial_value

        for i in range(1, len(weights)):
            date = weights.index[i]
            previous_date = weights.index[i-1]

            # Select top performer based on the criterion
            if criterion == 'performance':
                top_performer = performance_pivot.loc[previous_date].idxmax()
            elif criterion == 'risk':
                top_performer = risk_pivot.loc[previous_date].idxmin()
            elif criterion == 'risk_reward':
                risk_reward_ratio = performance_pivot.loc[previous_date] / risk_pivot.loc[previous_date]
                top_performer = risk_reward_ratio.idxmax()
            else:
                raise ValueError("Invalid criterion. Choose from 'performance', 'risk', 'risk_reward'.")

            # Reduce weights for all except the top performer
            for cik in weights.columns:
                if cik != top_performer:
                    weights.at[date, cik] = weights.at[previous_date, cik] * beta
                else:
                    weights.at[date, cik] = weights.at[previous_date, cik]
            
            # Normalize weights
            weights.loc[date] /= weights.loc[date].sum()

            # Create investment weights
            investment_weights = weights.copy()

            # Find top weights for each date and set others to 0
            max_weight = investment_weights.loc[date].max()
            investment_weights.loc[date] = investment_weights.loc[date].aaxpply(lambda x: x if x == max_weight else 0)
            
            # Normalize investment weights if there are multiple top weights
            if investment_weights.loc[date].sum() > 0:
                investment_weights.loc[date] /= investment_weights.loc[date].sum()

            # Calculate performance by investing based on investment weights
            cumulative_performance.at[date, 'Performance'] = (investment_weights.loc[date] * performance_pivot.loc[date]).sum()
            cumulative_performance.at[date, 'Risk'] = (investment_weights.loc[date] * risk_pivot.loc[date]).sum()
            cumulative_performance.at[date, 'Value'] = cumulative_performance.at[previous_date, 'Value'] * (1 + cumulative_performance.at[date, 'Performance'])

        cumulative_performance['Risk Reward Ratio'] = cumulative_performance['Performance'] / cumulative_performance['Risk']

        return cumulative_performance, weights
    
    # Optimize for the best beta
    # store the list of beta, final value
    list_beta_final_value = []
    
    for beta in beta_values:
        performance, weights = calculate_performance(df_grouped, beta)
        if criterion == 'performance':
            final_value = performance['Value'].iloc[-1]
        elif criterion == 'risk':
            final_value = -performance['Risk'].mean()
        elif criterion == 'risk_reward':
            final_value = performance['Risk Reward Ratio'].mean()
        list_beta_final_value.append((beta, final_value))

    best_beta, best_final_value = max(list_beta_final_value, key=lambda x: x[1])    
    beta_values, final_values = zip(*list_beta_final_value)

    if plot:
        print(f"Optimal beta: {best_beta}")
        print(f"Final value with optimal beta: {best_final_value}")

        # plot beta vs final value
        plt.plot(beta_values, final_values)
        plt.xlabel('Beta')
        plt.ylabel('Final Value')
        plt.title('Beta vs Final Value')
        plt.show()

    best_performance, best_weights = calculate_performance(df_grouped, best_beta)

    return best_performance, best_weights, beta_values, final_values


def weighted_majority_algorithm(df_grouped, beta_values, initial_value=1000, criterion='performance', plot = True):
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
    criterion : str
        The criterion to use for selecting the best expert ('performance', 'risk', 'risk_reward')
    
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
    beta_values : list
        List of beta values tested
    final_values : list
        List of final metric values for each beta value
    '''
    def calculate_performance(df_grouped, beta):
        df_grouped['new_fdate'] = pd.to_datetime(df_grouped['new_fdate'])
        
        performance_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_performance')
        risk_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_risk')

        # Initialize weights with equal weights for the first date
        initial_weights = 1 / performance_pivot.shape[1]
        weights = pd.DataFrame(initial_weights, index=performance_pivot.index, columns=performance_pivot.columns)

        # Initialize DataFrame to store cumulative performance
        cumulative_performance = pd.DataFrame(index=performance_pivot.index, columns=['Performance', 'Risk', 'Value'])
        cumulative_performance['Value'] = initial_value

        for i in range(1, len(weights)):
            date = weights.index[i]
            previous_date = weights.index[i-1]

            # Select top performer based on the criterion
            if criterion == 'performance':
                top_performer = performance_pivot.loc[previous_date].idxmax()
            elif criterion == 'risk':
                top_performer = risk_pivot.loc[previous_date].idxmin()
            elif criterion == 'risk_reward':
                risk_reward_ratio = performance_pivot.loc[previous_date] / risk_pivot.loc[previous_date]
                top_performer = risk_reward_ratio.idxmax()
            else:
                raise ValueError("Invalid criterion. Choose from 'performance', 'risk', 'risk_reward'.")

            # Reduce weights for all except the top performer
            for cik in weights.columns:
                if cik != top_performer:
                    weights.at[date, cik] = weights.at[previous_date, cik] * beta
                else:
                    weights.at[date, cik] = weights.at[previous_date, cik]
            
            # Normalize weights
            weights.loc[date] /= weights.loc[date].sum()

            # Calculate performance by investing based on investment weights
            cumulative_performance.at[date, 'Performance'] = (weights.loc[date] * performance_pivot.loc[date]).sum()
            cumulative_performance.at[date, 'Risk'] = (weights.loc[date] * risk_pivot.loc[date]).sum()
            cumulative_performance.at[date, 'Value'] = cumulative_performance.at[previous_date, 'Value'] * (1 + cumulative_performance.at[date, 'Performance'])

        cumulative_performance['Risk Reward Ratio'] = cumulative_performance['Performance'] / cumulative_performance['Risk']

        return cumulative_performance, weights
    
    # Optimize for the best beta
    # store the list of beta, final value
    list_beta_final_value = []
    
    for beta in beta_values:
        performance, weights = calculate_performance(df_grouped, beta)
        if criterion == 'performance':
            final_value = performance['Value'].iloc[-1]
        elif criterion == 'risk':
            final_value = -performance['Risk'].mean()
        elif criterion == 'risk_reward':
            final_value = performance['Risk Reward Ratio'].mean()
        list_beta_final_value.append((beta, final_value))

    best_beta, best_final_value = max(list_beta_final_value, key=lambda x: x[1])    
    beta_values, final_values = zip(*list_beta_final_value)

    if plot:
        print(f"Optimal beta: {best_beta}")
        print(f"Final value with optimal beta: {best_final_value}")

        # plot beta vs final value
        plt.plot(beta_values, final_values)
        plt.xlabel('Beta')
        plt.ylabel('Final Value')
        plt.title('Beta vs Final Value')
        plt.show()

    best_performance, best_weights = calculate_performance(df_grouped, best_beta)

    return best_performance, best_weights, beta_values, final_values

def weighted_majority_algorithm(df_grouped, beta_values, initial_value=1000, criterion='performance', plot = True):
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
    criterion : str
        The criterion to use for selecting the best expert ('performance', 'risk', 'risk_reward')
    
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
    beta_values : list
        List of beta values tested
    final_values : list
        List of final metric values for each beta value
    '''
    def calculate_performance(df_grouped, beta):
        df_grouped['new_fdate'] = pd.to_datetime(df_grouped['new_fdate'])
        
        performance_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_performance')
        risk_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_risk')

        # Initialize weights with equal weights for the first date
        initial_weights = 1 / performance_pivot.shape[1]
        weights = pd.DataFrame(initial_weights, index=performance_pivot.index, columns=performance_pivot.columns)

        # Initialize DataFrame to store cumulative performance
        cumulative_performance = pd.DataFrame(index=performance_pivot.index, columns=['Performance', 'Risk', 'Value'])
        cumulative_performance['Value'] = initial_value

        for i in range(1, len(weights)):
            date = weights.index[i]
            previous_date = weights.index[i-1]

            # Select top performer based on the criterion
            if criterion == 'performance':
                top_performer = performance_pivot.loc[previous_date].idxmax()
            elif criterion == 'risk':
                top_performer = risk_pivot.loc[previous_date].idxmin()
            elif criterion == 'risk_reward':
                risk_reward_ratio = performance_pivot.loc[previous_date] / risk_pivot.loc[previous_date]
                top_performer = risk_reward_ratio.idxmax()
            else:
                raise ValueError("Invalid criterion. Choose from 'performance', 'risk', 'risk_reward'.")

            # Reduce weights for all except the top performer
            for cik in weights.columns:
                if cik != top_performer:
                    weights.at[date, cik] = weights.at[previous_date, cik] * beta
                else:
                    weights.at[date, cik] = weights.at[previous_date, cik]
            
            # Normalize weights
            weights.loc[date] /= weights.loc[date].sum()

            # Calculate performance by investing based on investment weights
            cumulative_performance.at[date, 'Performance'] = (weights.loc[date] * performance_pivot.loc[date]).sum()
            cumulative_performance.at[date, 'Risk'] = (weights.loc[date] * risk_pivot.loc[date]).sum()
            cumulative_performance.at[date, 'Value'] = cumulative_performance.at[previous_date, 'Value'] * (1 + cumulative_performance.at[date, 'Performance'])

        cumulative_performance['Risk Reward Ratio'] = cumulative_performance['Performance'] / cumulative_performance['Risk']

        return cumulative_performance, weights
    
    # Optimize for the best beta
    # store the list of beta, final value
    list_beta_final_value = []
    
    for beta in beta_values:
        performance, weights = calculate_performance(df_grouped, beta)
        if criterion == 'performance':
            final_value = performance['Value'].iloc[-1]
        elif criterion == 'risk':
            final_value = -performance['Risk'].mean()
        elif criterion == 'risk_reward':
            final_value = performance['Risk Reward Ratio'].mean()
        list_beta_final_value.append((beta, final_value))

    best_beta, best_final_value = max(list_beta_final_value, key=lambda x: x[1])    
    beta_values, final_values = zip(*list_beta_final_value)

    if plot:
        print(f"Optimal beta: {best_beta}")
        print(f"Final value with optimal beta: {best_final_value}")

        # plot beta vs final value
        plt.plot(beta_values, final_values)
        plt.xlabel('Beta')
        plt.ylabel('Final Value')
        plt.title('Beta vs Final Value')
        plt.show()

    best_performance, best_weights = calculate_performance(df_grouped, best_beta)

    return best_performance, best_weights, beta_values, final_values


def randomized_weighted_majority_algorithm(df_grouped, beta_values, initial_value=1000, criterion='performance', plot = True):
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
    criterion : str
        The criterion to use for selecting the best expert ('performance', 'risk', 'risk_reward')
    
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
    beta_values : list
        List of beta values tested
    final_values : list
        List of final metric values for each beta value
    '''
    def calculate_performance(df_grouped, beta):
        df_grouped['new_fdate'] = pd.to_datetime(df_grouped['new_fdate'])
        
        performance_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_performance')
        risk_pivot = df_grouped.pivot(index='new_fdate', columns='cik', values='weighted_risk')

        # Initialize weights with equal weights for the first date
        initial_weights = 1 / performance_pivot.shape[1]
        weights = pd.DataFrame(initial_weights, index=performance_pivot.index, columns=performance_pivot.columns)

        # Initialize DataFrame to store cumulative performance
        cumulative_performance = pd.DataFrame(index=performance_pivot.index, columns=['Performance', 'Risk', 'Value', 'Risk Reward Ratio'])
        cumulative_performance['Value'] = initial_value
        cumulative_performance['Risk'][0] = risk_pivot[0].mean()
        cumulative_performance['Performance'][0] = performance_pivot[0].mean()
        cumulative_performance['Risk Reward Ratio'][0] = cumulative_performance['Performance'][0] / cumulative_performance['Risk'][0]

        for i in range(1, len(weights)):
            date = weights.index[i]
            previous_date = weights.index[i-1]

            # Select top performer based on the criterion
            if criterion == 'performance':
                top_performer = performance_pivot.loc[previous_date].idxmax()
            elif criterion == 'risk':
                top_performer = risk_pivot.loc[previous_date].idxmin()
            elif criterion == 'risk_reward':
                risk_reward_ratio = performance_pivot.loc[previous_date] / risk_pivot.loc[previous_date]
                top_performer = risk_reward_ratio.idxmax()
            else:
                raise ValueError("Invalid criterion. Choose from 'performance', 'risk', 'risk_reward'.")

            # Reduce weights for all except the top performer
            for cik in weights.columns:
                if cik != top_performer:
                    weights.at[date, cik] = weights.at[previous_date, cik] * beta
                else:
                    weights.at[date, cik] = weights.at[previous_date, cik]
            
            # Normalize weights
            weights.loc[date] /= weights.loc[date].sum()

            # Choose expert i randomly with probability w_i / W
            # where w_i is the weight of expert i and W is the sum of all weights
            selected_expert = np.random.choice(weights.columns, p=weights.loc[date])

            # Calculate performance by investing based on investment weights
            cumulative_performance.at[date, 'Performance'] = performance_pivot.loc[date, selected_expert]
            cumulative_performance.at[date, 'Risk'] = risk_pivot.loc[date, selected_expert]
            cumulative_performance.at[date, 'Value'] = cumulative_performance.at[previous_date, 'Value'] * (1 + cumulative_performance.at[date, 'Performance'])

        cumulative_performance['Risk Reward Ratio'] = cumulative_performance['Performance'] / cumulative_performance['Risk']

        return cumulative_performance, weights
    
    # Optimize for the best beta
    # store the list of beta, final value
    list_beta_final_value = []
    
    for beta in beta_values:
        performance, weights = calculate_performance(df_grouped, beta)
        if criterion == 'performance':
            final_value = performance['Value'].iloc[-1]
        elif criterion == 'risk':
            final_value = -performance['Risk'].mean()
        elif criterion == 'risk_reward':
            final_value = performance['Risk Reward Ratio'].mean()
        list_beta_final_value.append((beta, final_value))

    best_beta, best_final_value = max(list_beta_final_value, key=lambda x: x[1])    
    beta_values, final_values = zip(*list_beta_final_value)

    if plot:
        print(f"Optimal beta: {best_beta}")
        print(f"Final value with optimal beta: {best_final_value}")

        # plot beta vs final value
        plt.plot(beta_values, final_values)
        plt.xlabel('Beta')
        plt.ylabel('Final Value')
        plt.title('Beta vs Final Value')
        plt.show()

    best_performance, best_weights = calculate_performance(df_grouped, best_beta)

    return best_performance, best_weights, beta_values, final_values