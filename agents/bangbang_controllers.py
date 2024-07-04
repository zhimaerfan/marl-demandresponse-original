

class AlwaysOnController(object):
    """一直开 
    Bang bang controller taking deadband into account: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

    def __init__(self, agent_properties, config_dict, num_state = None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

    def act(self, obs):
        return True

class EvAlwaysOnController(object):
    """一直开, 根据额定功率,而不是最大功率
    Always on controller: always returns True for both active and reactive power."""

    def __init__(self, agent_properties, config_dict, num_state = None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

        self.transformer_max_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["max_power"]  # 充电桩的最大功率
        self.station_max_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["charging_stations"][0]["max_power"]
        self.station_rated_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["charging_stations"][0]["rated_power"]

        self.ratio_p = self.station_rated_power / self.station_max_power

    def act(self, obs):
        return [self.ratio_p, 1]

class DeadbandBangBangController(object):
    """ 考虑死区的BangBang,太热时开启，太冷时关闭，否则保持当前状态。
    Bang bang controller taking deadband into account: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

    def __init__(self, agent_properties, config_dict, num_state = None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

    def act(self, obs):
        obs = obs[self.id]
        house_temp = obs["house_temp"]
        house_target_temp = obs["house_target_temp"]
        house_deadband = obs["house_deadband"]
        hvac_turned_on = obs["hvac_turned_on"]

        if house_temp < house_target_temp-house_deadband/2:
            action = False
            # print("Too cold!")

        elif house_temp > house_target_temp+house_deadband/2:
            action = True
            # print("Too hot!")
        else:
            action = hvac_turned_on

        return action

class EvDeadbandBangBangController(object):
    """ 死区充电, 如果信号>总功率, soc<target_soc ± 5%, 额定非最大功率充电.
        Deadband Bang Bang controller for EVs: charges when signal is above total power and SOC is below target SOC, discharges otherwise."""

    def __init__(self, agent_properties, config_dict, num_state = None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]
        self.charging_deadband = config_dict["default_env_prop"]["cluster_prop"]["charging_deadband"]

        self.transformer_max_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["max_power"]  # 充电桩的最大功率
        self.station_max_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["charging_stations"][0]["max_power"]
        self.station_rated_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["charging_stations"][0]["rated_power"]

        self.ratio_p = self.station_rated_power / self.station_max_power

    def act(self, obs):
        obs = obs[self.id]
        battery_capacity = obs["battery_capacity"]
        # soc_diff_energy = obs["soc_diff_energy"]
        soc_target_energy = obs["soc_target_energy"]
        current_battery_energy = obs["current_battery_energy"]
        cluster_ev_active_power = obs["cluster_ev_active_power"]
        cluster_ev_reactive_power = obs["cluster_ev_reactive_power"]
        # current_ev_active_power = obs["current_ev_active_power"]
        # current_ev_reactive_power = obs["current_ev_reactive_power"]
        ev_active_reg_signal = obs["grid_ev_active_reg_signal"]
        ev_reactive_reg_signal = obs["grid_ev_reactive_reg_signal"]

        # 初始化有功和无功的动作值
        active_power_action = 0
        reactive_power_action = 0

        if battery_capacity > 0:  # 有EV
            # 计算target SOC范围，根据电池容量计算上下限
            lower_bound_soc = min(max(soc_target_energy - (battery_capacity * float(self.charging_deadband.strip('%')) / 100), 0), battery_capacity)
            upper_bound_soc = min(max(soc_target_energy + (battery_capacity * float(self.charging_deadband.strip('%')) / 100), 0), battery_capacity)

            # 判断是否充电或放电
            if ev_active_reg_signal > cluster_ev_active_power:
                if current_battery_energy < upper_bound_soc:
                    active_power_action = 1  # 充电
            elif ev_active_reg_signal < cluster_ev_active_power:
                if current_battery_energy > lower_bound_soc:
                    active_power_action = -1  # 放电

            if current_battery_energy < lower_bound_soc:
                active_power_action = 1  # 电量远不足,充电
            elif current_battery_energy > upper_bound_soc:
                active_power_action = -1  # 电量远超, 放电

        # 无论是否有EV，均根据无功信号来决定无功动作值
        if ev_reactive_reg_signal > cluster_ev_reactive_power:
            reactive_power_action = 1
        elif ev_reactive_reg_signal < cluster_ev_reactive_power:
            reactive_power_action = -1

        return [active_power_action * self.ratio_p, reactive_power_action]


class BangBangController(object):
    """
    不考虑死区的BangBang,太热时开启，太冷时关闭，否则保持当前状态, 精准控制。
    Cools when temperature is hotter than target (no interest for deadband). Limited on the hardware side by lockout (but does not know about it)
    """

    def __init__(self, agent_properties, config_dict, num_state = None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

    def act(self, obs):
        obs = obs[self.id]
        house_temp = obs["house_temp"]
        house_target_temp = obs["house_target_temp"]

        if house_temp <= house_target_temp:
            action = False

        elif house_temp > house_target_temp:
            action = True

        return action


class EvBangBangController(object):
    """不考虑死区的BangBang, 如果信号>总功率, soc<target_soc, 则额定非最大功率充电. 如果过了就放电，否则保持当前状态。
    Deadband Bang Bang controller for EVs: charges when signal is above total power and SOC is below target SOC, discharges otherwise."""

    def __init__(self, agent_properties, config_dict, num_state = None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]
        # self.charging_deadband = config_dict["default_env_prop"]["cluster_prop"]["charging_deadband"]
        self.charging_deadband = '0%'

        self.transformer_max_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["max_power"]  # 充电桩的最大功率
        self.station_max_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["charging_stations"][0]["max_power"]
        self.station_rated_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["charging_stations"][0]["rated_power"]

        self.ratio_p = self.station_rated_power / self.station_max_power

    def act(self, obs):
        obs = obs[self.id]
        battery_capacity = obs["battery_capacity"]
        # soc_diff_energy = obs["soc_diff_energy"]
        soc_target_energy = obs["soc_target_energy"]
        current_battery_energy = obs["current_battery_energy"]
        cluster_ev_active_power = obs["cluster_ev_active_power"]
        cluster_ev_reactive_power = obs["cluster_ev_reactive_power"]
        # current_ev_active_power = obs["current_ev_active_power"]
        # current_ev_reactive_power = obs["current_ev_reactive_power"]
        ev_active_reg_signal = obs["grid_ev_active_reg_signal"]
        ev_reactive_reg_signal = obs["grid_ev_reactive_reg_signal"]

        # 初始化有功和无功的动作值
        active_power_action = 0
        reactive_power_action = 0

        if battery_capacity > 0:  # 有EV
            # 计算target SOC范围，根据电池容量计算上下限
            lower_bound_soc = min(max(soc_target_energy - (battery_capacity * float(self.charging_deadband.strip('%')) / 100), 0), battery_capacity)
            upper_bound_soc = min(max(soc_target_energy + (battery_capacity * float(self.charging_deadband.strip('%')) / 100), 0), battery_capacity)

            # 判断是否充电或放电
            # if ev_active_reg_signal > cluster_ev_active_power:
            #     if current_battery_energy < upper_bound_soc:
            #         active_power_action = 1  # 充电
            # elif ev_active_reg_signal < cluster_ev_active_power:
            #     if current_battery_energy > lower_bound_soc:
            #         active_power_action = -1  # 放电

            if current_battery_energy < lower_bound_soc:
                active_power_action = 1  # 电量远不足,充电
            elif current_battery_energy > upper_bound_soc:
                active_power_action = -1  # 电量远超, 放电

        # 无论是否有EV，均根据无功信号来决定无功动作值
        if ev_reactive_reg_signal > cluster_ev_reactive_power:
            reactive_power_action = 1
        elif ev_reactive_reg_signal < cluster_ev_reactive_power:
            reactive_power_action = -1

        return [active_power_action * self.ratio_p, reactive_power_action]


class BasicController(object):
    """ 并非真正的bang bang控制器，但：过热时开启，过冷时关闭，否则保持当前状态
    Not really a bang bang controller but: turns on when too hot, turns off when too cold, sticks to current state otherwise"""

    def __init__(self, agent_properties, config_dict, num_state = None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]

    def act(self, obs):
        obs = obs[self.id]
        house_temp = obs["house_temp"]
        house_target_temp = obs["house_target_temp"]
        house_deadband = obs["house_deadband"]
        hvac_turned_on = obs["hvac_turned_on"]

        if house_temp < house_target_temp-house_deadband/2:
            action = False
            # print("Too cold!")

        elif house_temp > house_target_temp+house_deadband/2:
            action = True
            # print("Too hot!")
        else:
            action = hvac_turned_on

        return action

class EvBasicController(object):
    """ 并非真正的bang bang控制器，但：代码跟EvDeadbandBangBangController一样
    Not really a bang bang controller but: charges when SOC is below target, discharges when SOC is above target."""

    def __init__(self, agent_properties, config_dict, num_state = None):
        self.agent_properties = agent_properties
        self.id = agent_properties["id"]
        self.charging_deadband = config_dict["default_env_prop"]["cluster_prop"]["charging_deadband"]  # accurate、5%、uncontrol

        self.transformer_max_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["max_power"]  # 充电桩的最大功率
        self.station_max_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["charging_stations"][0]["max_power"]
        self.station_rated_power = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["charging_stations"][0]["rated_power"]

        self.ratio_p = self.station_rated_power / self.station_max_power

    def act(self, obs):
        obs = obs[self.id]
        battery_capacity = obs["battery_capacity"]
        # soc_diff_energy = obs["soc_diff_energy"]
        soc_target_energy = obs["soc_target_energy"]
        current_battery_energy = obs["current_battery_energy"]
        cluster_ev_active_power = obs["cluster_ev_active_power"]
        cluster_ev_reactive_power = obs["cluster_ev_reactive_power"]
        # current_ev_active_power = obs["current_ev_active_power"]
        # current_ev_reactive_power = obs["current_ev_reactive_power"]
        ev_active_reg_signal = obs["grid_ev_active_reg_signal"]
        ev_reactive_reg_signal = obs["grid_ev_reactive_reg_signal"]

        # 初始化有功和无功的动作值
        active_power_action = 0
        reactive_power_action = 0

        if battery_capacity > 0:  # 有EV
            # 计算target SOC范围，根据电池容量计算上下限
            lower_bound_soc = min(max(soc_target_energy - (battery_capacity * float(self.charging_deadband.strip('%')) / 100), 0), battery_capacity)
            upper_bound_soc = min(max(soc_target_energy + (battery_capacity * float(self.charging_deadband.strip('%')) / 100), 0), battery_capacity)

            # 判断是否充电或放电
            if ev_active_reg_signal > cluster_ev_active_power:
                if current_battery_energy < upper_bound_soc:
                    active_power_action = 1  # 充电
            elif ev_active_reg_signal < cluster_ev_active_power:
                if current_battery_energy > lower_bound_soc:
                    active_power_action = -1  # 放电

            if current_battery_energy < lower_bound_soc:
                active_power_action = 1  # 电量远不足,充电
            elif current_battery_energy > upper_bound_soc:
                active_power_action = -1  # 电量远超, 放电

        # 无论是否有EV，均根据无功信号来决定无功动作值
        if ev_reactive_reg_signal > cluster_ev_reactive_power:
            reactive_power_action = 1
        elif ev_reactive_reg_signal < cluster_ev_reactive_power:
            reactive_power_action = -1

        return [active_power_action * self.ratio_p, reactive_power_action]