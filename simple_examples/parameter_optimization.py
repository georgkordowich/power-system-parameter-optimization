"""
The goal of this script is to identify the uncertain parameter H (inertia constant) of a generator.
It is purposely written in a straight forward way to make it easy to understand.
Note: If you are not interested in the simulation process itself, but only the optimization,
collapse the functions load_flow(), differential(), algebraic() and do_sim().
"""
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import torch
from simple_examples.util.optimizer import CustomOptimizer

# Define parameters of the generator
S_n_gen = 2200
V_n_gen = 24
P_gen = 1998
V_soll_gen = 1
H_gen = 3.5
D_gen = 0
X_d_gen = 1.81
X_q_gen = 1.76
X_d_t_gen = 0.3
X_q_t_gen = 0.65
X_d_st_gen = 0.23
X_q_st_gen = 0.23
T_d0_t_gen = 8.0
T_q0_t_gen = 1
T_d0_st_gen = 0.03
T_q0_st_gen = 0.07

# Define Parameters of the Simple Exciter
v_ref_sexs = 1
K_sexs = 100
T_a_sexs = 4.0
T_b_sexs = 10.0
T_e_sexs = 0.1
E_min_sexs = -3
E_max_sexs = 3

# Define Parameters of the PSS (STAB1
H_lim_stab1 = 0.03

# Define parameters of the generator
S_n_IBB = 22000
V_n_IBB = 24
P_IBB = -1998
V_soll_IBB = 0.995
H_IBB = 3.5e7
D_IBB = 0
X_d_IBB = 1.8
X_q_IBB = 1.8
X_d_t_IBB = 0.3
X_q_t_IBB = 0.65
X_d_st_IBB = 0.23
X_q_st_IBB = 0.23
T_d0_t_IBB = 8
T_q0_t_IBB = 1
T_d0_st_IBB = 0.03
T_q0_st_IBB = 0.07

# Define system parameters
fn = 60
PF_n = 1
S_n_sys = 2200

# Define powerline parameters
X_line = 0.65


def load_flow():
    """
    Function performs a load flow using the Newton-Raphson Method. This is necessary to initialize the simulation
    :return: A list of initial values for the differential variables.
    """
    # For both nodes, the desired power is packed into an array
    # The desired power of the IBB is still multiplied by 10 here to get to per unit
    p_soll = torch.tensor([-P_gen / S_n_gen, -P_IBB / S_n_IBB * 10])
    q_soll = torch.tensor([0, 0])

    # Define the wanted voltage
    v_ibb = V_soll_IBB + 0j
    v_gen = V_soll_gen + 0j

    # Pack into one array
    v_ph = torch.tensor([v_gen, v_ibb], dtype=torch.complex128)

    # The admittance matrix is necessary for the load flow
    y_bus = torch.tensor([[1 / (1j * X_line), -1 / (1j * X_line)], [-1 / (1j * X_line), 1 / (1j * X_line)]],
                         dtype=torch.complex128)

    for ia in count(0):

        # To calculate the error, the power flow from the grid is calculated
        S_calc = v_ph * torch.conj(y_bus.matmul(v_ph))

        # The Jacobean matrix only has one value because the only PV bus is the generator
        # The Infinite Bus is the slack bus and therefore has no Jacobean value
        J = V_soll_gen ** 2 * -y_bus[0, 0].imag + S_calc.imag[0]

        # Here the error is calculated by subtracting the desired power of the generators from the calculated power
        # of the grid
        p_err = p_soll + S_calc.real
        q_err = q_soll + S_calc.imag

        error = p_err[0]  # only consider generator P because it is the only PV bus

        # get the partial derivative of the error with respect to the voltage angle in order to correct it by using
        # a gradient descent approach (Newton-Raphson)
        dx = error / J

        # The new voltage angle is calculated by subtracting the partial derivative from the old voltage angle
        delta_new = torch.angle(v_ph[0]) - dx

        # The new voltage is calculated by multiplying the desired voltage magnitude
        # with the exponential of the new voltage angle
        v_ph[0] = V_soll_gen * torch.exp(1j * delta_new)

        if abs(error) < 1e-8:
            # If the load flow converged, end the loop
            break

    ##################################################################################################################
    # Now, the load flow is finished, and the results can be used to initialize the differential equations
    ##################################################################################################################

    # First calculate the currents of the generator at the busbar.
    # Those currents can then be used to calculate all internal voltages.
    i_gen = torch.conj(S_calc[0] / v_ph[0])

    # Calculate the internal voltages and angle of the generator
    # Basically this is always U2 = U1 + jX * I
    e_q_gen = v_ph[0] + 1j * X_q_gen * i_gen
    angle_gen = torch.angle(e_q_gen)
    speed_gen = torch.tensor(0)

    i_dq_gen = i_gen * torch.exp(1j * (np.pi / 2 - angle_gen))
    i_d_gen = i_dq_gen.real
    i_q_gen = i_dq_gen.imag  # q-axis leading d-axis

    v_g_dq_gen = v_ph[0] * torch.exp(1j * (np.pi / 2 - angle_gen))
    v_d_gen = v_g_dq_gen.real
    v_q_gen = v_g_dq_gen.imag

    e_q_t_gen = v_q_gen + X_d_t_gen * i_d_gen
    e_d_t_gen = v_d_gen - X_q_t_gen * i_q_gen

    e_q_st_gen = v_q_gen + X_d_st_gen * i_d_gen
    e_d_st_gen = v_d_gen - X_q_st_gen * i_q_gen

    # Intialize global variables
    # Initialize global variables necessary in the differential equations.
    # Note this is not a nice solution, but it is easy to understand.
    global P_m_gen, bias
    P_m_gen = S_calc[0].real
    E_fd_gen = e_q_t_gen + i_d_gen * (X_d_gen - X_d_t_gen)

    # initialize simple exciter
    bias = E_fd_gen/K_sexs
    x_13_sexs = np.minimum(np.maximum(E_fd_gen, E_min_sexs), E_max_sexs)
    x_12_sexs = T_a_sexs * x_13_sexs / K_sexs - T_b_sexs * x_13_sexs / K_sexs

    # the pss can be initialized with zeros
    x_14_stab1 = torch.tensor(0)
    x_15_stab1 = torch.tensor(0)
    x_16_stab1 = torch.tensor(0)

    # Now repeat everything for the IBB
    i_ibb = torch.conj(S_calc[1]/10/v_ph[1])

    e_q_ibb = v_ph[1] + 1j * X_q_IBB * i_ibb
    angle_ibb = torch.angle(e_q_ibb)
    speed_ibb = torch.tensor(0)

    i_dq_ibb = i_ibb * torch.exp(1j * (np.pi / 2 - angle_ibb))
    i_d_ibb = i_dq_ibb.real
    i_q_ibb = i_dq_ibb.imag

    v_g_dq_ibb = v_ph[1] * torch.exp(1j * (np.pi / 2 - angle_ibb))
    v_d_ibb = v_g_dq_ibb.real
    v_q_ibb = v_g_dq_ibb.imag

    e_q_t_ibb = v_q_ibb + X_d_t_IBB * i_d_ibb
    e_d_t_ibb = v_d_ibb - X_q_t_IBB * i_q_ibb

    e_q_st_ibb = v_q_ibb + X_d_st_IBB * i_d_ibb
    e_d_st_ibb = v_d_ibb - X_q_st_IBB * i_q_ibb

    e_q_ibb = e_q_t_ibb + i_d_ibb * (X_d_IBB - X_d_t_IBB)

    # Initialize global variables
    global P_m_IBB, E_fd_IBB
    P_m_IBB = S_calc[1].real/10
    E_fd_IBB = e_q_ibb

    return torch.stack((speed_gen, angle_gen, e_q_t_gen, e_d_t_gen, e_q_st_gen, e_d_st_gen, speed_ibb, angle_ibb, e_q_t_ibb, e_d_t_ibb, e_q_st_ibb, e_d_st_ibb, x_12_sexs, x_13_sexs, x_14_stab1, x_15_stab1, x_16_stab1))



def differential(x, y):
    """
    Calculates all the differential equations.
    :param x: Differential variables as a list or a tensor
    :param y: Algebraic variables as a list or a tensor
    :return: The new dx of all differential variables as a tensor
    """
    # First, unpack all variables
    P_e_gen = y[0]
    v_bb_gen = y[1]
    i_d_gen = y[2]
    i_q_gen = y[3]

    P_e_ibb = y[4]
    v_bb_ibb = y[5]
    i_d_ibb = y[6]
    i_q_ibb = y[7]

    omega_gen = x[0]
    delta_gen = x[1]
    e_q_t_gen = x[2]
    e_d_t_gen = x[3]
    e_q_st_gen = x[4]
    e_d_st_gen = x[5]

    omega_ibb = x[6]
    delta_ibb = x[7]
    e_q_t_ibb = x[8]
    e_d_t_ibb = x[9]
    e_q_st_ibb = x[10]
    e_d_st_ibb = x[11]

    x_12_sexs = x[12]
    x_13_sexs = x[13]

    x_14_stab1 = x[14]
    x_15_stab1 = x[15]
    x_16_stab1 = x[16]

    E_fd_IBB = 1.06042

    T_m_gen = P_m_gen / (1 + omega_gen)
    T_m_IBB = P_m_IBB / (1 + omega_ibb)

    # Here go forward in the STAB1 PSS
    gain_out_stab1 = K_stab1 * omega_gen
    washout_out_stab1 = (gain_out_stab1 - x_14_stab1) / T_w_stab1
    lead_lag1_out_stab1 = 1 / T_3_stab1 * (T_1_stab1 * washout_out_stab1 - x_15_stab1)
    lead_lag2_out_stab1 = 1/T_4_stab1*(T_2_stab1 * lead_lag1_out_stab1 - x_16_stab1)
    v_ref_stab1 = torch.minimum(torch.maximum(lead_lag2_out_stab1.real, torch.tensor(-H_lim_stab1)), torch.tensor(H_lim_stab1))


    # First go forward in the simple exciter
    # Todo: Replace the bias with transducer stuff
    v_err_sexs = v_ref_sexs - abs(v_bb_gen) + bias + v_ref_stab1
    v_reg = 1/T_b_sexs*(T_a_sexs*v_err_sexs - x_12_sexs)
    v_reg_gain = K_sexs*v_reg
    E_fd_gen = torch.minimum(torch.maximum(x_13_sexs.real, torch.tensor(E_min_sexs)), torch.tensor(E_max_sexs))

    # Calculate the differential variables of the generator
    dx0 = 1 / (2 * H_gen) * (T_m_gen - P_e_gen)
    dx1 = omega_gen * 2 * np.pi * fn
    dx2 = 1 / T_d0_t_gen * (E_fd_gen - e_q_t_gen - i_d_gen * (X_d_gen - X_d_t_gen))
    dx3 = 1 / T_q0_t_gen * (-e_d_t_gen + i_q_gen * (X_q_gen - X_q_t_gen))
    dx4 = 1 / T_d0_st_gen * (e_q_t_gen - e_q_st_gen - i_d_gen * (X_d_t_gen - X_d_st_gen))
    dx5 = 1 / T_q0_st_gen * (e_d_t_gen - e_d_st_gen + i_q_gen * (X_q_t_gen - X_q_st_gen))

    # Calculate the differential variables of the IBB
    dx6 = 1 / (2 * H_IBB) * (T_m_IBB - P_e_ibb - D_IBB * omega_ibb)
    dx7 = omega_ibb * 2 * np.pi * fn
    dx8 = 1 / T_d0_t_IBB * (E_fd_IBB - e_q_t_ibb - i_d_ibb * (X_d_IBB - X_d_t_IBB))
    dx9 = 1 / T_q0_t_IBB * (-e_d_t_ibb + i_q_ibb * (X_q_IBB - X_q_t_IBB))
    dx10 = 1 / T_d0_st_IBB * (e_q_t_ibb - e_q_st_ibb - i_d_ibb * (X_d_t_IBB - X_d_st_IBB))
    dx11 = 1 / T_q0_st_IBB * (e_d_t_ibb - e_d_st_ibb + i_q_ibb * (X_q_t_IBB - X_q_st_IBB))

    # Simple exciter
    dx12 = (T_a_sexs / T_b_sexs - 1) * v_err_sexs - 1 / T_b_sexs * x_12_sexs
    dx13 = 1 / T_e_sexs * (v_reg_gain - x_13_sexs)
    dx13 = torch.tensor(0) if (x_13_sexs.real <= E_min_sexs and dx13.real < 0) or (x_13_sexs.real >= E_max_sexs and dx13.real > 0) else dx13

    # STAB1 PSS
    dx14 = washout_out_stab1
    dx15 = (T_1_stab1 / T_3_stab1 - 1) * washout_out_stab1 - 1 / T_3_stab1 * x_15_stab1
    dx16 = (T_2_stab1 / T_4_stab1 - 1) * lead_lag1_out_stab1 - 1 / T_4_stab1 * x_16_stab1

    return torch.stack((dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12, dx13, dx14, dx15, dx16))


def algebraic(x, y, sc_on):
    """
    Calculates all algebraic equations including current injections. The sc_on parameter is used to determine if the
    short circuit is on or off, and thus influences the admittance matrix.
    :param x: Differential variables as a list or a tensor.
    :param y: Algebraic variables as a list or a tensor.
    :param sc_on: Boolean value to determine if the short circuit is on or off.
    :return: The new tensor of algebraic variables.
    """
    P_e_gen = y[0]
    v_bb_gen = y[1]
    i_d_gen = y[2]
    i_q_gen = y[3]

    P_e_ibb = y[4]
    v_bb_ibb = y[5]
    i_d_ibb = y[6]
    i_q_ibb = y[7]

    omega_gen = x[0]
    delta_gen = x[1]
    e_q_t_gen = x[2]
    e_d_t_gen = x[3]
    e_q_st_gen = x[4]
    e_d_st_gen = x[5]

    omega_ibb = x[6]
    delta_ibb = x[7]
    e_q_t_ibb = x[8]
    e_d_t_ibb = x[9]
    e_q_st_ibb = x[10]
    e_d_st_ibb = x[11]

    # admittance matrix
    if sc_on:
        y_adm = torch.tensor([[(-1j / X_d_st_gen - 1j / X_line) + 1000000, 1j / X_line],
                              [1j / X_line, -1j / X_line - 1j / (X_d_st_IBB / 10)]])
    else:
        y_adm = torch.tensor([[-1j / X_d_st_gen - 1j / X_line, 1j / X_line],
                              [1j / X_line, -1j / X_line - 1j / (X_d_st_IBB / 10)]])

    y_inv = torch.linalg.inv(y_adm)

    d_gen = torch.exp(1j * (delta_gen - np.pi / 2))
    q_gen = torch.exp(1j * delta_gen)

    d_ibb = torch.exp(1j * (delta_ibb - np.pi / 2))
    q_ibb = torch.exp(1j * delta_ibb)

    # Calculate current injections
    i_d_gen = e_q_st_gen / (1j * X_d_st_gen) * q_gen
    i_q_gen = e_d_st_gen / (1j * X_q_st_gen) * d_gen
    i_inj_gen = i_d_gen + i_q_gen

    i_d_ibb = e_q_st_ibb / (1j * X_d_st_IBB) * q_ibb
    i_q_ibb = e_d_st_ibb / (1j * X_q_st_IBB) * d_ibb
    i_inj_ibb = (i_d_ibb + i_q_ibb) * 10

    # Calculate voltages at the bus
    v_bb_gen = y_inv[0, 0] * i_inj_gen + y_inv[0, 1] * i_inj_ibb
    v_bb_ibb = y_inv[1, 0] * i_inj_gen + y_inv[1, 1] * i_inj_ibb

    # Calculate power injections
    P_e_gen = (v_bb_gen * torch.conj((e_q_st_gen * q_gen + e_d_st_gen * d_gen - v_bb_gen) / (1j * X_d_st_gen))).real

    P_e_ibb = (v_bb_ibb * torch.conj((e_q_st_ibb * q_ibb + e_d_st_ibb * d_ibb - v_bb_ibb) / (1j * X_d_st_IBB))).real

    i_gen = (e_q_st_gen * q_gen + e_d_st_gen * d_gen - v_bb_gen) / (1j * X_d_st_gen)
    i_ibb = (e_q_st_ibb * q_ibb + e_d_st_ibb * d_ibb - v_bb_ibb) / (1j * X_d_st_IBB)

    i_d_gen = (i_gen * torch.exp(1j * (np.pi / 2 - delta_gen))).real
    i_q_gen = (i_gen * torch.exp(1j * (np.pi / 2 - delta_gen))).imag

    i_d_ibb = (i_ibb * torch.exp(1j * (np.pi / 2 - delta_ibb))).real
    i_q_ibb = (i_ibb * torch.exp(1j * (np.pi / 2 - delta_ibb))).imag

    return torch.stack((P_e_gen, v_bb_gen, i_d_gen, i_q_gen, P_e_ibb, v_bb_ibb, i_d_ibb, i_q_ibb))


def do_sim():
    # First the load flow is executed to get the initial values
    # Note: this is done a bit quick and dirty, as global variables are also initialized here
    # Use the initial values from the load flow as initial values for the dynamic simulation
    # Those values correspond omegas, deltas, e_q_t, e_d_t, e_q_st, e_d_st of the generator and the IBB
    x0 = load_flow()

    # The algebraic variables are initialized to 1 but will be overwritten immediately
    y0 = [1, 1, 1, 1, 1, 1, 1, 1]
    y0 = algebraic(x0, y0, False)

    # Define the simulation time
    time = torch.linspace(0, 10, 2001)
    x_result = []

    for k, t_step in enumerate(time):
        # Activate or deactivate the short circuit
        if 1 <= t_step <= 1.05:
            sc_on = True
        else:
            sc_on = False

        # Use a modified version of Euler's method to solve the differential equations (i.e. integrate them step by
        # step)
        dxdt_0 = differential(x0, y0)
        x_1_guess = x0 + dxdt_0 * (time[1] - time[0])
        y_1 = algebraic(x_1_guess, y0, sc_on)
        dxdt_1 = differential(x_1_guess, y_1)
        x_1 = x0 + (dxdt_0 + dxdt_1) / 2 * (time[1] - time[0])

        # Update values
        x0 = x_1
        y0 = y_1

        # Save the results
        x_result.append(x_1)

    res = torch.vstack(x_result)

    return time, res


if __name__ == '__main__':
    ##################################################################################################################
    # Note: Here the actual optimization is done
    ##################################################################################################################

    results = []
    # Define the parameters to optimize and their initial values
    K_stab1 = torch.tensor(40, requires_grad=True, dtype=torch.complex128)
    T_w_stab1 = torch.tensor(11, requires_grad=True, dtype=torch.complex128)
    T_1_stab1 = torch.tensor(0.08, requires_grad=True, dtype=torch.complex128)
    T_3_stab1 = torch.tensor(0.1, requires_grad=True, dtype=torch.complex128)
    T_2_stab1 = torch.tensor(0.5, requires_grad=True, dtype=torch.complex128)
    T_4_stab1 = torch.tensor(0.05, requires_grad=True, dtype=torch.complex128)

    optimizer = CustomOptimizer([K_stab1, T_w_stab1, T_1_stab1, T_3_stab1, T_2_stab1, T_4_stab1], decay=1)

    last_loss = 100000

    for i in range(50):
        # reset the gradients
        optimizer.zero_grad()

        # then run the simulation
        t, result = do_sim()

        # only consider the results after the short circuit for the loss function
        res2 = result[110:, 0].real

        # Use the MAE as loss function
        loss = torch.mean(torch.abs(res2))
        print('loss: ', float(loss))

        # Calculate the gradients by traversing the computational graph backwards
        loss.backward()

        # Update all parameters using the custom optimizer
        optimizer.step()

        print('New Params: K_stab1: {}, T_w_stab1: {}, T_1_stab1: {}, T_2_stab1: {}, T_3_stab1: {}, T_4_stab1: {}'.format(
            K_stab1.data, T_w_stab1.data, T_1_stab1.data, T_3_stab1.data, T_2_stab1.data, T_4_stab1.data))

        plt.plot(t.detach().numpy().real, result[:, 0].real.detach().numpy(), label=r'$\Delta \omega_{gen, sim}$', linewidth=1.0)

        plt.xlabel('Time (s)')
        plt.ylabel(r'$\Delta \omega$ (pu)')

        plt.tight_layout()
        plt.legend()
        plt.show()

        results.append(result[:, 0].detach().numpy())

        if abs(last_loss - loss) < 1e-4:
            # stop the optimization if the loss function does not change anymore
            break
        else:
            last_loss = loss

    print('Final Params: K_stab1: {}, T_w_stab1: {}, T_1_stab1: {}, T_2_stab1: {}, T_3_stab1: {}, T_4_stab1: {}'.format(
        K_stab1.data, T_w_stab1.data, T_1_stab1.data, T_3_stab1.data, T_2_stab1.data, T_4_stab1.data))