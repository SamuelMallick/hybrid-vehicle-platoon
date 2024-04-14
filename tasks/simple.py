def leader_state_1(self, ep_len, N):
        """Leader trajectory with constant velocity"""
        leader_state = np.zeros((2, ep_len + N + 1))
        leader_speed = 20
        leader_initial_pos = 3000
        leader_state[:, [0]] = np.array([[leader_initial_pos], [leader_speed]])
        for k in range(ep_len + N):
            leader_state[:, [k + 1]] = leader_state[:, [k]] + self.ts * np.array(
                [[leader_speed], [0]]
            )
        return leader_state

    def leader_state_2(self, ep_len, N):
        """Leader trajectory with speed up and slow down to same initial speed."""
        leader_state = np.zeros((2, ep_len + N + 1))
        leader_speed = 20
        leader_initial_pos = 600
        leader_state[:, [0]] = np.array([[leader_initial_pos], [leader_speed]])
        for k in range(int(ep_len / 4)):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        leader_speed = 30
        for k in range(int(ep_len / 4), int(1 * ep_len / 2)):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        leader_speed = 20
        for k in range(int(1 * ep_len / 2), ep_len + N):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        return leader_state

    def leader_state_3(self, ep_len, N):
        """Leader trajectory with speed up, then slow down to slower than initial, then speed back up to initial speed."""
        leader_state = np.zeros((2, ep_len + N + 1))
        leader_speed = 20
        leader_initial_pos = 600
        leader_state[:, [0]] = np.array([[leader_initial_pos], [leader_speed]])
        for k in range(int(ep_len / 4)):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        leader_speed = 30
        for k in range(int(ep_len / 4), int(1 * ep_len / 2)):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        leader_speed = 10
        for k in range(int(1 * ep_len / 2), int(3 * ep_len / 4)):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        leader_speed = 20
        for k in range(int(3 * ep_len / 4), ep_len + N):
            leader_state[:, [k + 1]] = np.array(
                [[leader_state[0, k]], [leader_speed]]
            ) + self.ts * np.array([[leader_speed], [0]])
        return leader_state