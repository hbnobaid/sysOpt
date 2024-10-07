**Integrating Socio-Economic Data into the Energy Management RL Framework**

---

**Dear User,**

Thank you for providing the socio-economic data for each house. I'll help you incorporate this data into the RL framework to adjust the comfort and reward parameters accordingly. This integration will allow the model to personalize energy management strategies based on the socio-economic characteristics of each household.

---

### **Understanding the Socio-Economic Data**

The socio-economic data includes various attributes such as:

- **Age of the Respondent**
- **Number of Occupants**
- **Family Monthly Income**
- **Pets**
- **House Typology**
- **Educational Level**
- **Heating and Cooling Systems**
- **Appliance Usage Patterns**

**Objective:**

- Use this data to inform and adjust comfort preferences and reward parameters in the RL framework.
- Personalize energy management strategies to better align with the needs and preferences of each household.

---

### **Incorporating Socio-Economic Data into the RL Framework**

**Approach:**

1. **Load Socio-Economic Data:**

   - Read the socio-economic data CSV files for each house.
   - Extract relevant attributes that influence comfort preferences and energy usage sensitivity.

2. **Adjust Comfort Preferences:**

   - **Temperature Preferences:**
     - Older adults may prefer warmer indoor temperatures.
     - Adjust the desired temperature range based on the age of occupants.
   - **Cost Sensitivity:**
     - Households with higher income may prioritize comfort over cost savings.
     - Adjust the weight of energy cost versus comfort penalties in the reward function.
   - **Pet Presence:**
     - Households with pets may have specific comfort requirements.
     - Adjust comfort parameters to account for pets.

3. **Modify Reward Function:**

   - Incorporate socio-economic factors into the calculation of comfort penalties and energy cost importance.
   - Personalize the reward function for each household based on their socio-economic profile.

4. **Update the Environment and Training Code:**

   - Modify the `LHEMSEnv` class to include socio-economic data.
   - Update the federated training loop to pass socio-economic data to each environment.

---

### **Updated Code**

Below is the updated code that integrates socio-economic data into the RL framework.

```python
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Function to load and preprocess electric data
def load_and_preprocess_electric_data(house_dir):
    # List CSV files in the electric data directory
    electric_dir = os.path.join(house_dir, 'Electric_data')
    csv_files = [f for f in os.listdir(electric_dir) if f.endswith('.csv')]
    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(electric_dir, csv_file)
        df = pd.read_csv(file_path)
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        dfs.append(df)
    # Concatenate all months
    electric_df = pd.concat(dfs)
    # Resample to 1-minute intervals
    electric_df = electric_df.resample('1T').mean()
    # Forward-fill small gaps (up to 2 missing intervals, i.e., 30 seconds)
    electric_df = electric_df.ffill(limit=2)
    # Handle missing values
    electric_df = electric_df.dropna()
    return electric_df

# Function to load and preprocess environmental data
def load_and_preprocess_environmental_data(house_dir):
    # Environmental data directory
    env_dir = os.path.join(house_dir, 'Environmental_data')
    csv_files = [f for f in os.listdir(env_dir) if f.endswith('.csv')]
    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(env_dir, csv_file)
        df = pd.read_csv(file_path)
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        dfs.append(df)
    # Concatenate all months
    env_df = pd.concat(dfs)
    # Resample to 1-minute intervals
    env_df = env_df.resample('1T').interpolate()
    # Forward-fill small gaps (up to 60 minutes)
    env_df = env_df.ffill(limit=60)
    # Handle missing values
    env_df = env_df.dropna()
    return env_df

# Function to load socio-economic data
def load_socio_economic_data(house_dir):
    # Socio-economic data file
    socio_file = os.path.join(house_dir, 'Sociodemographic', 'socioeconomic_data.csv')
    df = pd.read_csv(socio_file)
    # Convert to dictionary
    socio_data = df.to_dict(orient='records')[0]  # Assuming one record per house
    return socio_data

# Local Home Energy Management System Environment
class LHEMSEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
    }

    def __init__(self, electric_data, env_data, appliances_metadata, comfort_prefs, socio_data):
        super().__init__()
        self.electric_data = electric_data
        self.env_data = env_data
        self.appliances_metadata = appliances_metadata
        self.socio_data = socio_data  # Socio-economic data

        self.appliance_names = [col for col in electric_data.columns if col not in ['V', 'A', 'P_agg', 'issues']]
        self.num_appliances = len(self.appliance_names)
        self.current_time_index = 0
        self.max_time_steps = len(electric_data)

        # Adjust comfort preferences based on socio-economic data
        self.comfort_prefs = self.adjust_comfort_prefs(comfort_prefs)

        # Define action space: Each appliance can be ON or OFF
        self.action_space = gym.spaces.MultiDiscrete([2]*self.num_appliances)

        # Define observation space
        # For each appliance: [Normalized Power, Desired State]
        # Environmental data: [Internal Temp, Internal Humidity, External Temp, External Humidity]
        # Time Features: [Hour, Day of Week]
        obs_low = [0.0]*(2*self.num_appliances) + [-50.0]*4 + [0.0, 0.0]
        obs_high = [1.0]*(2*self.num_appliances) + [50.0]*4 + [1.0, 1.0]
        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32
        )

    def adjust_comfort_prefs(self, base_prefs):
        adjusted_prefs = base_prefs.copy()

        # Example adjustments based on socio-economic data

        # Adjust temperature preferences based on age
        age = self.socio_data.get('Age of the respondent', 35)
        if age >= 65:
            # Older individuals may prefer warmer temperatures
            adjusted_prefs['temperature_range'] = [22.0, 27.0]
        elif age <= 25:
            # Younger individuals may prefer cooler temperatures
            adjusted_prefs['temperature_range'] = [18.0, 23.0]
        else:
            # Default temperature range
            adjusted_prefs['temperature_range'] = base_prefs.get('temperature_range', [20.0, 25.0])

        # Adjust cost sensitivity based on income
        income_bracket = self.socio_data.get('Family monthly income', 'Medium')
        if income_bracket == 'High':
            # Less sensitive to energy cost
            adjusted_prefs['cost_weight'] = 0.5
            adjusted_prefs['comfort_weight'] = 1.5
        elif income_bracket == 'Low':
            # More sensitive to energy cost
            adjusted_prefs['cost_weight'] = 1.5
            adjusted_prefs['comfort_weight'] = 0.5
        else:
            # Default weights
            adjusted_prefs['cost_weight'] = 1.0
            adjusted_prefs['comfort_weight'] = 1.0

        # Adjust comfort preferences if pets are present
        pets = self.socio_data.get('Pets', 'No pets')
        if pets != 'No pets':
            # Increase comfort weight
            adjusted_prefs['comfort_weight'] += 0.5

        return adjusted_prefs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time_index = 0
        self.done = False
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self._take_action(action)
        observation = self._get_obs()
        reward = self._calculate_reward(action)
        terminated = self._is_terminated()
        truncated = False
        info = {}

        self.current_time_index += 1
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        obs = []
        timestamp = self.electric_data.index[self.current_time_index]

        # Appliance observations
        for appliance_name in self.appliance_names:
            power = self.electric_data[appliance_name].iloc[self.current_time_index]
            # Normalize power based on appliance metadata
            max_power = self.appliances_metadata.get(appliance_name, {}).get('cutoff', 1.0)
            normalized_power = power / max_power if max_power > 0 else 0.0
            desired_state = 1 if normalized_power > 0.1 else 0
            obs.extend([normalized_power, desired_state])

        # Environmental data
        internal_temp = self.env_data['internal_temperature'].iloc[self.current_time_index]
        internal_humidity = self.env_data['internal_humidity'].iloc[self.current_time_index]
        external_temp = self.env_data['external_temperature'].iloc[self.current_time_index]
        external_humidity = self.env_data['external_humidity'].iloc[self.current_time_index]
        obs.extend([internal_temp, internal_humidity, external_temp, external_humidity])

        # Time features
        hour = timestamp.hour / 23.0  # Normalize to [0,1]
        day_of_week = timestamp.dayofweek / 6.0  # Normalize to [0,1]
        obs.extend([hour, day_of_week])

        return np.array(obs, dtype=np.float32)

    def _take_action(self, action):
        # Update appliance states based on action
        self.actions = action

    def _calculate_reward(self, action):
        total_reward = 0
        timestamp = self.electric_data.index[self.current_time_index]

        # Environmental comfort parameters
        internal_temp = self.env_data['internal_temperature'].iloc[self.current_time_index]
        desired_temp_range = self.comfort_prefs.get('temperature_range', [20.0, 25.0])
        temp_penalty = self.comfort_prefs.get('temp_penalty', 5.0)

        # Cost and comfort weights
        cost_weight = self.comfort_prefs.get('cost_weight', 1.0)
        comfort_weight = self.comfort_prefs.get('comfort_weight', 1.0)

        for idx, appliance_name in enumerate(self.appliance_names):
            power = self.electric_data[appliance_name].iloc[self.current_time_index]
            max_power = self.appliances_metadata.get(appliance_name, {}).get('cutoff', 1.0)
            normalized_power = power / max_power if max_power > 0 else 0.0
            desired_state = 1 if normalized_power > 0.1 else 0
            actual_state = action[idx]
            price = 1.0  # Modify as needed

            # Negative electric cost
            energy_cost = -cost_weight * price * normalized_power * actual_state

            # Comfort penalty
            comfort_penalty = 0

            # Appliance-specific comfort preferences
            appliance_prefs = self.comfort_prefs.get(appliance_name, {})
            if desired_state == 1 and actual_state == 0:
                comfort_penalty += comfort_weight * appliance_prefs.get('penalty', 10.0)

            # For AC units, consider temperature comfort
            if appliance_name.startswith('ac') and actual_state == 0:
                if internal_temp > desired_temp_range[1]:
                    comfort_penalty += comfort_weight * temp_penalty * (internal_temp - desired_temp_range[1])
                elif internal_temp < desired_temp_range[0]:
                    comfort_penalty += comfort_weight * temp_penalty * (desired_temp_range[0] - internal_temp)

            # Total reward for the appliance
            reward = energy_cost - comfort_penalty
            total_reward += reward

        return total_reward

    def _is_terminated(self):
        return self.current_time_index >= self.max_time_steps - 1

    def render(self):
        pass

    def close(self):
        pass

# Function to train local models for each LHEMS
def train_local_model(env, global_params=None, total_timesteps=10000):
    # Initialize the local model
    model = PPO('MlpPolicy', env, verbose=0)
    if global_params:
        model.policy.load_state_dict(global_params)
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    # Return the trained model parameters
    return model.policy.state_dict()

# Function to aggregate global model parameters
def aggregate_global_model(local_models_params):
    global_params = {}
    num_models = len(local_models_params)
    for key in local_models_params[0].keys():
        # Average the parameters across all local models
        global_params[key] = sum([model_params[key] for model_params in local_models_params]) / num_models
    return global_params

# Federated training loop
def federated_training(lhems_envs, num_rounds=5, local_timesteps=10000):
    # Initialize local models
    local_models = [None] * len(lhems_envs)
    global_params = None

    for round in range(num_rounds):
        print(f"--- Federated Training Round {round+1}/{num_rounds} ---")
        local_models_params = []
        # Local training at each LHEMS
        for idx, env in enumerate(lhems_envs):
            print(f"Training local model for LHEMS {idx+1}")
            local_params = train_local_model(env, global_params=global_params, total_timesteps=local_timesteps)
            local_models_params.append(local_params)
        # Global aggregation at GS
        global_params = aggregate_global_model(local_models_params)
        print("Global model aggregated and updated.")

    # Return the final global model parameters
    return global_params

# Example usage
if __name__ == "__main__":
    # Directory containing house folders
    data_dir = "./Plegma Dataset/CleanDataset"
    house_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.startswith('House')]

    # Base comfort preferences
    base_comfort_prefs = {
        'ac_1': {'penalty': 5.0},
        'ac_2': {'penalty': 5.0},
        'boiler': {'penalty': 10.0},
        'fridge': {'penalty': 8.0},
        'washing_machine': {'penalty': 6.0},
        'temperature_range': [20.0, 25.0],  # Default desired indoor temperature range
        'temp_penalty': 5.0,  # Penalty per degree outside desired range
        'cost_weight': 1.0,   # Default cost weight
        'comfort_weight': 1.0, # Default comfort weight
    }

    # Appliances metadata (e.g., from appliances_metadata.csv)
    appliances_metadata = {
        'ac_1': {'cutoff': 2000.0},
        'ac_2': {'cutoff': 2000.0},
        'boiler': {'cutoff': 1500.0},
        'fridge': {'cutoff': 300.0},
        'washing_machine': {'cutoff': 1000.0},
        # Add more appliances as needed
    }

    # Create LHEMS environments for each house
    lhems_envs = []
    for house_dir in house_dirs:
        # Load electric data
        electric_data = load_and_preprocess_electric_data(house_dir)
        # Load environmental data
        env_data = load_and_preprocess_environmental_data(house_dir)
        # Load socio-economic data
        socio_data = load_socio_economic_data(house_dir)
        # Create environment
        env = LHEMSEnv(electric_data, env_data, appliances_metadata, base_comfort_prefs, socio_data)
        lhems_envs.append(env)

    # Run federated training
    final_global_params = federated_training(lhems_envs, num_rounds=5, local_timesteps=10000)
    print("Federated training completed.")

    # Save the final global model parameters
    print("Final global model parameters are ready for use.")
```

---

### **Explanation of Changes**

#### **1. Loading Socio-Economic Data**

- **Function `load_socio_economic_data`**:
  - Reads the socio-economic data CSV file for each house.
  - Converts the data into a dictionary for easy access.
  - Assumes the data is stored in a folder named `Sociodemographic` within each house directory.

#### **2. Adjusting Comfort Preferences**

- **Method `adjust_comfort_prefs` in `LHEMSEnv`**:
  - Takes the base comfort preferences and adjusts them based on the socio-economic data.
  - **Age-Based Adjustments**:
    - Alters the desired temperature range based on the age of the respondent.
  - **Income-Based Adjustments**:
    - Modifies the weights of energy cost and comfort in the reward function.
    - Households with higher income are less sensitive to cost and may prioritize comfort.
  - **Pet Presence Adjustments**:
    - Increases the comfort weight if pets are present in the household.

#### **3. Modifying the Reward Function**

- **Method `_calculate_reward` in `LHEMSEnv`**:
  - Incorporates the adjusted comfort preferences into the reward calculation.
  - Uses the `cost_weight` and `comfort_weight` to balance the importance of energy cost and comfort penalties.
  - Applies the adjusted temperature range for temperature-related comfort penalties.

#### **4. Updating the Environment Initialization**

- The `LHEMSEnv` class now accepts `socio_data` as an argument.
- Adjusts comfort preferences during initialization based on the socio-economic data.

#### **5. Main Execution**

- **Loading Socio-Economic Data**:
  - In the main loop, loads the socio-economic data for each house using the `load_socio_economic_data` function.
  - Passes the socio-economic data to the environment during creation.

---

### **Notes and Considerations**

- **Data Paths**:
  - Ensure that the socio-economic data CSV files are correctly placed in the `Sociodemographic` folder within each house directory.
  - Update the file paths if your directory structure differs.

- **Socio-Economic Data Format**:
  - The code assumes that the socio-economic data CSV file contains key-value pairs that can be converted into a dictionary.
  - Modify the `load_socio_economic_data` function if the data format is different.

- **Adjustments Based on Data**:
  - The adjustments made in the `adjust_comfort_prefs` method are examples.
  - You can refine the logic to better reflect the influence of socio-economic factors on comfort preferences.

- **Scalability**:
  - The code is designed to handle multiple houses with different socio-economic profiles.

- **Privacy Considerations**:
  - Socio-economic data is used locally within each LHEMS.
  - The federated learning framework ensures that personal data is not shared between LHEMSs or with the global server.

---

### **Conclusion**

By integrating socio-economic data into the RL framework, we've enhanced the personalization of the energy management strategies for each household. The model now adjusts comfort preferences and reward parameters based on factors such as age, income, and pet ownership, leading to more tailored and effective energy management solutions.

---

**Please let me know if you have any questions or need further assistance with the code or its implementation. I'm here to support you in successfully integrating socio-economic data into your energy management RL framework.**

---

**Best regards, and I wish you continued success with your project!**

---