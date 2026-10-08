
### Costs: paper

|                                           | ann_return   | ann_vol   |   sharpe | max_drawdown   | hit_ratio   |   skew |   trades_per_day |
|:------------------------------------------|:-------------|:----------|---------:|:---------------|:------------|-------:|-----------------:|
| ('Train', 'Ext. 1: + band/VWAP stop')     | 4.2%         | 6.5%      |     0.66 | 11.6%          | 41.8%       |   0.68 |             0.86 |
| ('Test', 'Ext. 1: + band/VWAP stop')      | 6.7%         | 6.6%      |     1.02 | 10.0%          | 45.7%       |   1.72 |             0.92 |
| ('Train', 'Ext. 2: + vol targeting')      | 13.3%        | 14.6%     |     0.93 | 25.1%          | 41.8%       |   2.38 |             0.86 |
| ('Test', 'Ext. 2: + vol targeting')       | 15.8%        | 14.2%     |     1.1  | 22.3%          | 45.7%       |   2.07 |             0.92 |
| ('Train', 'SPY buy & hold')               | 18.4%        | 17.5%     |     1.05 | 33.8%          | 56.4%       |  -0.88 |                  |
| ('Test', 'SPY buy & hold')                | 12.2%        | 17.3%     |     0.75 | 24.5%          | 53.5%       |   0.34 |                  |
| ('Train', 'ML: logistic (1x)')            | -0.3%        | 9.6%      |     0.02 | 23.6%          | 52.0%       |  -1.01 |             1.21 |
| ('Test', 'ML: logistic (1x)')             | 5.7%         | 10.3%     |     0.59 | 12.8%          | 51.3%       |  -1.35 |             1.32 |
| ('Train', 'ML: logistic + vol targeting') | 3.9%         | 20.6%     |     0.29 | 33.1%          | 52.0%       |  -0.22 |             1.21 |
| ('Test', 'ML: logistic + vol targeting')  | 13.3%        | 20.6%     |     0.71 | 20.5%          | 51.3%       |   0.32 |             1.32 |

### Sub-periods (paper costs)

|                                                                              | ann_return   | ann_vol   |   sharpe | max_drawdown   |   sharpe_t |
|:-----------------------------------------------------------------------------|:-------------|:----------|---------:|:---------------|-----------:|
| ('Test, before publication (2022-01 - 2024-04)', 'Ext. 1: + band/VWAP stop') | 14.2%        | 7.3%      |     1.86 | 4.7%           |       2.83 |
| ('Test, before publication (2022-01 - 2024-04)', 'ML: logistic (1x)')        | 6.6%         | 10.3%     |     0.67 | 12.8%          |       1.02 |
| ('Test, before publication (2022-01 - 2024-04)', 'SPY buy & hold')           | 3.9%         | 18.6%     |     0.3  | 24.5%          |       0.46 |
| ('After publication (2024-05 - today)', 'Ext. 1: + band/VWAP stop')          | 0.0%         | 5.8%      |     0.03 | 10.0%          |       0.05 |
| ('After publication (2024-05 - today)', 'ML: logistic (1x)')                 | 4.8%         | 10.3%     |     0.51 | 9.4%           |       0.79 |
| ('After publication (2024-05 - today)', 'SPY buy & hold')                    | 20.8%        | 16.0%     |     1.26 | 18.8%          |       1.96 |

Correlation of daily returns ML vs Ext. 1 (test period): -0.03
