import numpy as np
import os
from collections import deque, defaultdict

class Network(object):
    def __init__(self, caseName, g=9.81):
        self.caseName = caseName
        self.g = g
        self.network = np.loadtxt(caseName + '/network', dtype=int)

        with open(caseName+'/input', "r") as file:
            lines = file.readlines()
        self.CFL = float(lines[1][:-1])
        self.RBFtype = int(lines[4][:-1])
        self.time_integrator = int(lines[7][:-1])
        self.simEndTime = float(lines[10][:-1])
        self.diffLim = float(lines[13][:-1])
        self.time = 0
        self.printStep = float(lines[16][:-1])

        file.close()

        self.num_segments = sum(
            1 for d in os.listdir(caseName) if d.startswith("segment")
            and os.path.isdir(os.path.join(caseName, d)))
        self.segments = self.load_segments()

        if self.num_segments == 1:
            self.calcOrder = [0]
            self.segUpstreamInfo = []
            self.segDownstreamInfo = []
        else:
            self.queSegments()

    def load_segments(self):
        """Initialize and store SingleChannel objects for each segment."""
        segments = {}
        for i in range(self.num_segments):
            segments[i] = SingleChannel(self.caseName, i, self.RBFtype)
        return segments

    def queSegments(self):
        '''write an algorithm which reads network and creates an array of calculation order of segments,
        will return self.calcOrder
        also, self.segUpstreamInfo and self.segDownstreamInfo
        these two dicts will keep the segment ids immediately upstream of current segment.'''
        self.segUpstreamInfo = defaultdict(list)
        self.segDownstreamInfo = defaultdict(list)
        in_degree = defaultdict(int)  # Store number of upstream segments for each node
        all_nodes = set()

        # Read network connections (assuming self.network stores tuples of (upstream, downstream))
        for upstream, downstream in self.network:
            self.segUpstreamInfo[downstream].append(upstream)
            self.segDownstreamInfo[upstream].append(downstream)
            in_degree[downstream] += 1
            all_nodes.update([upstream, downstream])

        # Find segments with zero in-degree (no upstream dependencies)
        zero_in_degree = deque([node for node in all_nodes if in_degree[node] == 0])

        # Process in topological order
        self.calcOrder = []
        while zero_in_degree:
            segment = zero_in_degree.popleft()
            self.calcOrder.append(segment)

            for downstream in self.segDownstreamInfo[segment]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    zero_in_degree.append(downstream)

        # Convert to numpy array
        self.calcOrder = np.array(self.calcOrder)

    def solve(self):

        if self.time_integrator == 0:
            while self.time < self.simEndTime:

                dt = 1e8
                for i in range(self.num_segments):
                    self.segments[i].update_params(self.diffLim)
                    u = self.segments[i].Q / self.segments[i].area
                    ws = np.sqrt(self.segments[i].cele * self.segments[i].h)
                    dt_arr = self.segments[i].geo['dx'] / (np.abs(u) + np.abs(ws))
                    dt = min(dt, np.min(dt_arr))
                dt = self.CFL * dt
                self.time += dt
                print(self.time)

                for i in self.calcOrder:
                    self.segments[i].read_upstream_Q(self.time)
                    self.segments[i].solveSeg_fwEuler(dt)
                    for j in self.segDownstreamInfo:
                        self.update_junction_Q(self.segments[i].Q[-1], j)
                for i in self.calcOrder[::-1]:
                    self.segments[i].solveSeg_h()
                    for j in self.segUpstreamInfo:
                        self.update_junction_h(self.segments[i].h[0], j)

    def update_junction_Q(self, Q, segId):
        """Update the boundary condition Q for downstream segments."""

        file_path = f"{self.caseName}/segment{segId}/geo/boundary_Q"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Create a new entry
        new_entry = np.array([[self.time, Q]])

        # Check if the file exists and has data
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            # Load existing data
            existing_data = np.loadtxt(file_path)

            # Ensure it's a 2D array even if there's only one row
            if existing_data.ndim == 1:
                existing_data = existing_data.reshape(1, -1)

            # Append new entry
            updated_data = np.vstack((existing_data, new_entry))
        else:
            # If file doesn't exist or is empty, start fresh
            updated_data = new_entry

        # Save back to file
        np.savetxt(file_path, updated_data, fmt="%.6f")

    def update_junction_h(self, h, segId):
        """Update the boundary condition Q for downstream segments."""

        # Define the file path
        file_path = f"{self.caseName}/segment{segId}/geo/boundary_h"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Create a new entry
        new_entry = np.array([[self.time, h]])

        # Check if the file exists and has data
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            # Load existing data
            existing_data = np.loadtxt(file_path)

            # Ensure it's a 2D array even if there's only one row
            if existing_data.ndim == 1:
                existing_data = existing_data.reshape(1, -1)

            # Append new entry
            updated_data = np.vstack((existing_data, new_entry))
        else:
            # If file doesn't exist or is empty, start fresh
            updated_data = new_entry

        # Save back to file
        np.savetxt(file_path, updated_data, fmt="%.6f")

class SingleChannel(object):
    """Radial Basis Function Collocation Method for 1D diffusive wave equation."""

    def __init__(self, caseName, segmentNo, rbf_type, g=9.81):
        self.caseName = caseName
        self.segmentNo = segmentNo
        self.g = g
        self.load_geometry()
        self.nodeNo = len(self.geo['nodes'])
        self.initialize_conditions()
        self.RBFtype = rbf_type
        self.compute_RBF_matrix()

    def load_geometry(self):
        """Load geometry-related data for the segment."""
        self.geom_path = f"{self.caseName}/segment{self.segmentNo}/geo/"
        self.geo = {
            'nodes': np.loadtxt(self.geom_path + 'nodes'),
            'slopes': np.loadtxt(self.geom_path + 'slopes'),
            'xsInfo': np.loadtxt(self.geom_path + 'xsInfo', dtype=int),
            'mannings_n': np.loadtxt(self.geom_path + 'mannings_n')
        }
        gdx = np.zeros_like(self.geo['nodes'])
        gdx[1:-1] = (self.geo['nodes'][1:-1] - self.geo['nodes'][:-2]) / 2 + (self.geo['nodes'][2:] - self.geo['nodes'][1:-1]) / 2
        gdx[0], gdx[-1] = gdx[1], gdx[-2]
        self.geo.update({'dx': gdx})
        self.boundary_Q = np.loadtxt(self.geom_path + 'boundary_Q')
        self.boundary_h = np.loadtxt(self.geom_path + 'boundary_h')


    def initialize_conditions(self):
        run_path = self.caseName + '/segment' + str(self.segmentNo) + '/run/'
        self.Q = np.loadtxt(run_path+'0/Q.csv')
        self.h = np.loadtxt(run_path + '0/h.csv')
        self.lat = np.zeros_like(self.Q)
        self.area = np.zeros_like(self.Q)
        self.cele = np.zeros_like(self.Q)
        self.diffu = np.zeros_like(self.Q)
        self.Sf = np.zeros_like(self.Q)
        self.I = np.eye(self.nodeNo)


    def compute_RBF_matrix(self):
        self.f = np.zeros((self.nodeNo, self.nodeNo))
        self.fx = np.zeros_like(self.f)
        self.fxx = np.zeros_like(self.f)

        if self.RBFtype == 0:
            self.buildTPS(2)
        elif self.RBFtype == 1:
            self.buildMQ(0)

    def buildMQ(self, shapeParameter = 0): # 0 for 4 rmin, 1 for .815rav
        xdif = np.zeros_like(self.f)
        for i in range(self.nodeNo):
            for j in range(self.nodeNo):
                xdif[i, j] = self.geo['nodes'][i] - self.geo['nodes'][j]
        r = np.abs(xdif)
        rmin = np.min(r[0,1:])
        if shapeParameter == 0:
            c = 4 * rmin

        self.hsys = self.fx
        self.hsys[-1,:] = self.f[-1,:]
        inv_hsys = np.linalg.pinv(self.hsys)
        self.hsys = np.matmul(self.f, inv_hsys)

        self.Qsys = self.f
        self.Qsys[-1, :] = self.fx[-1, :]
        inv_Qsys = np.linalg.pinv(self.Qsys)
        self.Qsys = np.matmul(inv_Qsys, self.f)
        invF = np.linalg.pinv(self.f)
        self.fx_invF = np.matmul(self.fx, invF)
        self.fxx_invF = np.matmul(self.fxx, invF)

    def buildTPS(self, beta=2):
        for i in range(self.nodeNo):
            for j in range(self.nodeNo):
                if i != j:
                    r = np.abs(self.geo['nodes'][i] - self.geo['nodes'][j])
                    self.f[i,j] = r ** beta * np.log(r)
                    self.fx[i,j] = (self.geo['nodes'][i] - self.geo['nodes'][j]) * r ** (beta - 2) * (
                                beta * np.log(r) + 1)
                    self.fxx[i,j] = r ** (beta - 2) * (beta * np.log(r) + 1) + (
                            self.geo['nodes'][i] - self.geo['nodes'][j]) ** 2 * r ** (beta - 4) * (
                                              2 * (beta - 1) + beta * (beta - 2) * np.log(r))

        self.hsys = self.fx
        self.hsys[-1,:] = self.f[-1,:]
        inv_hsys = np.linalg.pinv(self.hsys)
        self.hsys = np.matmul(self.f, inv_hsys)

        self.Qsys = self.f
        self.Qsys[-1, :] = self.fx[-1, :]
        inv_Qsys = np.linalg.pinv(self.Qsys)
        self.Qsys = np.matmul(inv_Qsys, self.f)
        invF = np.linalg.pinv(self.f)
        self.fx_invF = np.matmul(self.fx, invF)
        self.fxx_invF = np.matmul(self.fxx, invF)


    def update_bc(self, time):
        self.Q[0] = self.interpBC_Q(time)
        self.h[0] = self.interpBC_h(time)
        self.lat = self.readLateral(time)

    def readLateral(self, time):
        pass

    def update_params(self, diffLim):
        for i in range(self.nodeNo):
            wetPerim, self.area[i] = self.interp_wet_area(i)
            R = self.area[i] / wetPerim
            self.Sf[i] = self.geo['mannings_n'][i] ** 2 / (self.area[i]* R ** (2 / 3)) ** 2 * self.Q[i] ** 2
            self.cele[i] = 5 * self.Sf[i] ** .3 * self.Q[i] ** .4 / 3 / self.area[i] ** .4 * self.h[i] ** .4 / self.geo['mannings_n'][i] ** .6
            self.diffu[i] = min(diffLim, np.abs(self.Q[i]) * self.h[i] / self.area[i] / 2 / self.Sf[i])

    def solveSeg_fwEuler(self, dt):
        '''Calculate new Q'''
        adv = np.matmul(self.I * (self.cele), np.matmul(self.fx_invF, self.Q))
        diff = np.matmul(self.I * self.diffu, np.matmul(self.fxx_invF, self.Q))
        lat = self.cele * self.lat
        '''Euler here'''
        self.Q[1:-1] +=  dt * (-adv[1:-1] + diff[1:-1] - lat[1:-1])
        self.Q[-1] += self.f[-1:,:]

    def solveSeg_h(self):
        '''Calculate new h'''
        RHS = self.geo['slopes'] - self.Sf
        RHS[-1] = self.h[-1]
        self.h[:-1] = np.matmul(self.hsys[:-1,:], RHS)

    def interp_wet_area(self, i):
        h = self.h[i]
        xsNo = self.geo['xsInfo'][i]
        xs = np.loadtxt(self.caseName + '/segment' + str(self.segmentNo) + '/geo/xs' + str(xsNo))

        # Find the points submerged by water level h
        below_h = xs[:,0] <= h

        # Interpolate water surface points
        x1_interp = np.interp(h, xs[:,0], xs[:,1])
        x2_interp = np.interp(h, xs[:,0], xs[:,2])

        # Compute Area using Trapezoidal rule
        area = 0
        wp = xs[0,2] - xs[0,1]
        for i, tr in enumerate(below_h[1:]):
            if tr:
                area += (xs[i+1,0] - xs[i,0]) * ((xs[i, 2] - xs[i, 1]) + (xs[i+1, 2] - xs[i+1, 1])) / 2
                wp += np.sqrt((xs[i+1,1] - xs[i,1])**2 + (xs[i+1,0] - xs[i,0])**2)
                wp += np.sqrt((xs[i+1, 2] - xs[i, 2]) ** 2 + (xs[i+1, 0] - xs[i, 0]) ** 2)
            else:
                area += (h - xs[i,0]) * ((xs[i, 2] - xs[i, 1]) + (x2_interp - x1_interp)) / 2
                wp += np.sqrt((x1_interp - xs[i, 1])**2 + (h - xs[i,0])**2)
                wp += np.sqrt((x2_interp - xs[i, 2]) ** 2 + (h - xs[i, 0]) ** 2)
                break

        return wp, area

    def read_upstream_Q(self, time):
        Q_boundaries = np.loadtxt(self.geom_path + 'boundary_Q')
        times = Q_boundaries[:, 0]  # Extract time column
        Q_values = Q_boundaries[:, 1]  # Extract Q column

        # Interpolate Q for the given time
        Q_interp = np.interp(time, times, Q_values)

        return Q_interp

    def read_downstream_h(self, time):
        h_boundaries = np.loadtxt(self.geom_path + 'boundary_h')
        times = h_boundaries[:, 0]  # Extract time column
        h_values = h_boundaries[:, 1]  # Extract Q column

        # Interpolate Q for the given time
        h_interp = np.interp(time, times, h_values)

        return h_interp