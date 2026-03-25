use hashbrown::HashMap;
use hashbrown::HashSet;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::collections::VecDeque;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub type CommunityId = u32;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct Graph<N, E> {
    _nodes: Vec<N>,
    _edges: Vec<EdgeInfo<E>>,
    /// Adjacency list: _connections[i] is a Vec of (neighbor_node, edge_id) pairs,
    /// sorted by neighbor_node for O(log k) lookup.
    _connections: Vec<Vec<(usize, usize)>>,
    _total_weight: f32,
}

impl<N, E> Default for Graph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct EdgeInfo<E> {
    pub edge_data: E,
    pub weight: f32,
    pub id: usize,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub enum Community {
    L1Community(HashSet<usize> /* nodes */),
    LNCommunity(Vec<Community> /* communities */),
}

impl Community {
    pub fn collect_nodes(&self, f: &impl Fn(usize)) {
        match self {
            Community::L1Community(nodes) => {
                for &node in nodes.iter() {
                    f(node);
                }
            }
            Community::LNCommunity(communities) => {
                for community in communities.iter() {
                    community.collect_nodes(f);
                }
            }
        }
    }
}

pub trait ModularityOptimizer {
    fn is_converged(&mut self, previous: f32, current: f32) -> bool;
    fn get_parallel_threshold(&self) -> usize;
}

pub struct TrivialModularityOptimizer {
    /// Parallel scale
    ///
    /// If the node count exceeds this value, the optimizer will use parallel
    /// optimization.
    pub parallel_scale: usize,

    /// Tolerance for modularity change
    ///
    /// If the modularity change is less than this value, the optimizer will
    /// consider the optimization converged.
    pub tol: f32,
}

impl ModularityOptimizer for TrivialModularityOptimizer {
    #[inline]
    fn is_converged(&mut self, previous: f32, current: f32) -> bool {
        current - previous < self.tol
    }

    #[inline]
    fn get_parallel_threshold(&self) -> usize {
        self.parallel_scale
    }
}

impl<N, E> Graph<N, E> {
    pub const fn new() -> Self {
        Self {
            _nodes: Vec::new(),
            _edges: Vec::new(),
            _connections: Vec::new(),
            _total_weight: 0.0,
        }
    }

    pub fn node_data_slice(&self) -> &[N] {
        &self._nodes
    }

    pub fn add_node(&mut self, node_data: N) -> usize {
        let id = self._nodes.len();
        self._nodes.push(node_data);
        self._connections.push(Vec::new());
        id
    }

    pub fn add_edge(&mut self, n1: usize, n2: usize, edge_data: E, weight: f32) -> Option<usize> {
        if n1 == n2 {
            return None;
        }

        match self._connections[n1].binary_search_by_key(&n2, |&(nb, _)| nb) {
            Ok(pos) => {
                let edge_id = self._connections[n1][pos].1;
                self._edges[edge_id].weight += weight;
                self._total_weight += weight;
                self._edges[edge_id].edge_data = edge_data;
                Some(edge_id)
            }
            Err(pos1) => {
                let edge_id = self._edges.len();
                self._edges.push(EdgeInfo {
                    edge_data,
                    weight,
                    id: edge_id,
                });
                self._connections[n1].insert(pos1, (n2, edge_id));
                let pos2 = self._connections[n2].partition_point(|&(nb, _)| nb < n1);
                self._connections[n2].insert(pos2, (n1, edge_id));
                self._total_weight += weight;
                Some(edge_id)
            }
        }
    }

    pub fn count_nodes(&self) -> usize {
        self._nodes.len()
    }

    pub fn try_get_edge_between(&self, n1: usize, n2: usize) -> Option<&EdgeInfo<E>> {
        match self._connections[n1].binary_search_by_key(&n2, |&(nb, _)| nb) {
            Ok(pos) => Some(&self._edges[self._connections[n1][pos].1]),
            Err(_) => None,
        }
    }
}

pub struct LocalMove {
    pub node: usize,
    pub community: u32,
}

type CommunityAssignments = HashMap<usize, CommunityId>;

trait MaybeLocalMove {
    fn get(&self) -> Option<&LocalMove>;
}

impl MaybeLocalMove for LocalMove {
    #[inline]
    fn get(&self) -> Option<&LocalMove> {
        Some(self)
    }
}

impl MaybeLocalMove for () {
    #[inline]
    fn get(&self) -> Option<&LocalMove> {
        None
    }
}

impl<N: Send + Sync, E: Send + Sync> Graph<N, E> {
    pub fn initial_community(&self) -> CommunityAssignments {
        let count_nodes = self.count_nodes();
        let mut assignments = HashMap::with_capacity(count_nodes);
        for i in 0..CommunityId::try_from(count_nodes).expect("nodes must be less than u32::MAX") {
            assignments.insert(i as usize, i);
        }
        assignments
    }

    #[inline]
    pub fn compute_modularity(&self, assignments: &CommunityAssignments) -> f32 {
        self._compute_modularity_impl(assignments, ())
    }

    #[inline]
    pub fn compute_modularity_with_local_move(
        &self,
        assignments: &CommunityAssignments,
        local_move: LocalMove,
    ) -> f32 {
        self._compute_modularity_impl(assignments, local_move)
    }

    #[inline]
    fn _compute_modularity_impl(
        &self,
        assignments: &CommunityAssignments,
        local_move: impl MaybeLocalMove,
    ) -> f32 {
        let m = self._total_weight;
        let node_count: usize = self.count_nodes();
        let mut q = 0.0;

        macro_rules! get_assignment {
            ($i:ident) => {
                match local_move.get() {
                    None => assignments[&$i],
                    Some(local_move) => {
                        if local_move.node == $i {
                            local_move.community
                        } else {
                            assignments[&$i]
                        }
                    }
                }
            };
        }

        // Precompute weighted degrees: k[i] = sum of incident edge weights.
        let weighted_degrees: Vec<f32> = (0..node_count)
            .map(|i| {
                self._connections[i]
                    .iter()
                    .map(|&(_, edge_id)| self._edges[edge_id].weight)
                    .sum()
            })
            .collect();

        for i in 0..node_count {
            let assigni = get_assignment!(i);
            let conn_i = &self._connections[i];
            let ki = weighted_degrees[i];
            for j in (i + 1)..node_count {
                let assignj = get_assignment!(j);
                if assigni != assignj {
                    continue;
                }

                let kj = weighted_degrees[j];

                match conn_i.binary_search_by_key(&j, |&(nb, _)| nb) {
                    Ok(pos) => {
                        let edge_id = conn_i[pos].1;
                        let edge_ij_weight = self._edges[edge_id].weight;
                        q += edge_ij_weight - (ki * kj) / (m + m);
                    }
                    Err(_) => {
                        q += -ki * kj / (m + m);
                    }
                }
            }
        }

        return q / m;
    }

    fn optimize_modularity(
        &self,
        assignments: &mut CommunityAssignments,
        optimizer: &mut impl ModularityOptimizer,
    ) {
        let node_count = self.count_nodes();

        // Precompute weighted degrees once.
        let weighted_degrees: Vec<f32> = (0..node_count)
            .map(|i| {
                self._connections[i]
                    .iter()
                    .map(|&(_, eid)| self._edges[eid].weight)
                    .sum()
            })
            .collect();

        // sigma_tot[c] = sum of weighted degrees of all nodes in community c.
        let mut sigma_tot: HashMap<CommunityId, f32> = HashMap::new();
        for (&node, &community) in assignments.iter() {
            *sigma_tot.entry(community).or_insert(0.0) += weighted_degrees[node];
        }

        let parallel_threshold = optimizer.get_parallel_threshold();

        if node_count < parallel_threshold {
            // Sequential path: apply each move immediately (online updates).
            // This avoids the cycling that batch-then-apply can produce.
            loop {
                let mut any_moved = false;
                for i in 0..node_count {
                    if let Some(local_move) =
                        self.fast_local_move(i, assignments, &weighted_degrees, &sigma_tot)
                    {
                        let old_community = assignments[&local_move.node];
                        let new_community = local_move.community;
                        let k = weighted_degrees[local_move.node];
                        *sigma_tot.get_mut(&old_community).unwrap() -= k;
                        *sigma_tot.entry(new_community).or_insert(0.0) += k;
                        assignments.insert(local_move.node, new_community);
                        any_moved = true;
                    }
                }
                if !any_moved {
                    break;
                }
            }
        } else {
            // Parallel path: collect moves in parallel (read-only snapshot of state),
            // apply serially, check convergence via global modularity to detect cycling.
            let mut batch_moving: boxcar::Vec<LocalMove> = boxcar::Vec::new();
            let mut previous_modularity = self.compute_modularity(assignments);

            loop {
                (0..node_count).into_par_iter().for_each(|node| {
                    if let Some(local_move) =
                        self.fast_local_move(node, assignments, &weighted_degrees, &sigma_tot)
                    {
                        batch_moving.push(local_move);
                    }
                });

                if batch_moving.is_empty() {
                    break;
                }

                for (_, local_move) in batch_moving.iter() {
                    let old_community = assignments[&local_move.node];
                    let new_community = local_move.community;
                    let k = weighted_degrees[local_move.node];
                    *sigma_tot.get_mut(&old_community).unwrap() -= k;
                    *sigma_tot.entry(new_community).or_insert(0.0) += k;
                    assignments.insert(local_move.node, new_community);
                }

                batch_moving.clear();

                let current_modularity = self.compute_modularity(assignments);
                if optimizer.is_converged(previous_modularity, current_modularity) {
                    break;
                }
                previous_modularity = current_modularity;
            }
        }
    }

    /// Compute the best community to move `node` to using incremental delta modularity.
    /// Returns None if no move improves modularity.
    fn fast_local_move(
        &self,
        node: usize,
        assignments: &CommunityAssignments,
        weighted_degrees: &[f32],
        sigma_tot: &HashMap<CommunityId, f32>,
    ) -> Option<LocalMove> {
        let m = self._total_weight;
        let k_i = weighted_degrees[node];
        let c_i = assignments[&node];
        let sigma_i = sigma_tot.get(&c_i).copied().unwrap_or(0.0);

        // Sum edge weights from node to each neighboring community.
        let mut community_weights: HashMap<CommunityId, f32> = HashMap::new();
        for &(neighbor, edge_id) in &self._connections[node] {
            let c_j = assignments[&neighbor];
            *community_weights.entry(c_j).or_insert(0.0) += self._edges[edge_id].weight;
        }

        let k_i_to_i = community_weights.get(&c_i).copied().unwrap_or(0.0);

        // Score of staying in c_i (after removing i, sigma drops by k_i).
        let baseline = k_i_to_i / m - k_i * (sigma_i - k_i) / (2.0 * m * m);

        let mut best_community = c_i;
        let mut best_gain: f32 = 0.0;

        for (&c_j, &k_i_to_j) in &community_weights {
            if c_j == c_i {
                continue;
            }
            let sigma_j = sigma_tot.get(&c_j).copied().unwrap_or(0.0);
            let score = k_i_to_j / m - k_i * sigma_j / (2.0 * m * m);
            let gain = score - baseline;
            if gain > best_gain {
                best_gain = gain;
                best_community = c_j;
            }
        }

        if best_community != c_i {
            Some(LocalMove {
                node,
                community: best_community,
            })
        } else {
            None
        }
    }

    fn refine(&self, assignments: &CommunityAssignments) -> Vec<HashSet<usize>> {
        // this is the community assignments by louvain
        // each community might get split into multiple communities
        // if there are partitions that are not connected to each other
        let mut communities_by_louvain: Vec<HashSet<usize>> = vec![];

        // fill and relabel
        {
            let mut relabel: HashMap<u32, u32> = HashMap::new();

            let mut assure_relabel_community =
                |communities_by_louvain: &mut Vec<HashSet<usize>>, louvain_community: u32| -> u32 {
                    match relabel.get(&louvain_community) {
                        Some(community) => *community,
                        None => {
                            let relabel_community = relabel.len() as u32;
                            #[cfg(debug_assertions)]
                            {
                                debug_assert!(
                                    relabel_community as usize == communities_by_louvain.len()
                                );
                            }
                            relabel.insert(louvain_community, relabel_community);
                            communities_by_louvain.push(HashSet::new());
                            relabel_community
                        }
                    }
                };

            for (&node, &louvain_community) in assignments.iter() {
                let relabel_community =
                    assure_relabel_community(&mut communities_by_louvain, louvain_community);
                communities_by_louvain[relabel_community as usize].insert(node);
            }
        }

        // XXX: parallelize?
        // validate the inner connections in each community
        let mut i = 0;
        while i < communities_by_louvain.len() {
            let community = &communities_by_louvain[i];

            if community.len() == 1 {
                i += 1;
                continue;
            }

            debug_assert!(community.len() > 1);

            let mut left_members = community.clone();
            let mut queue = VecDeque::new();

            queue.push_back(*community.iter().next().unwrap());

            while let Some(node) = queue.pop_front() {
                let newly_visited = left_members.remove(&node);
                if !newly_visited {
                    // already visited, skip
                    continue;
                }

                let neighbors = &self._connections[node];

                for &(neighbor, _) in neighbors.iter() {
                    if !community.contains(&neighbor) {
                        // the sub-community shall not get connected via this node
                        continue;
                    }

                    queue.push_back(neighbor);
                }
            }

            if left_members.is_empty() {
                // all members are connected, no need to split
            } else {
                let community = &mut communities_by_louvain[i];
                // split the community into two
                for _ in community.extract_if(|node| left_members.contains(node)) {
                    /* force consume to perform the elimination */
                }
                // optimization:
                // we already know that `left_members` are connected, and
                // if `left_members` are larger, we swap them as `communities_by_louvain[i]`
                // so that it will not be resolved in the later rounds.
                if left_members.len() > community.len() {
                    std::mem::swap(&mut left_members, community);
                }
                communities_by_louvain.push(left_members);
            }
            i += 1;
        }

        communities_by_louvain
    }

    pub fn leiden(
        &self,
        max_iter: Option<usize>,
        optimizer: &mut impl ModularityOptimizer,
    ) -> Graph<Community, ()> {
        let mut high_level_graph: Graph<Community, ()>;
        {
            let g = leiden_l1(&self, optimizer);
            let node_count_g1 = g.count_nodes();
            let g = leiden_ln(g, optimizer);
            let node_count_g2 = g.count_nodes();
            if node_count_g2 == node_count_g1 {
                return g;
            }
            high_level_graph = g;
        }

        let mut count = high_level_graph.count_nodes();
        let mut previous: usize;

        if let Some(mut max_iter) = max_iter {
            loop {
                previous = count;
                high_level_graph = leiden_ln(high_level_graph, optimizer);
                count = high_level_graph.count_nodes();
                if (previous == count) | (max_iter == 0) {
                    break;
                }
                max_iter -= 1;
            }
        } else {
            loop {
                previous = count;
                high_level_graph = leiden_ln(high_level_graph, optimizer);
                count = high_level_graph.count_nodes();
                if previous == count {
                    break;
                }
            }
        }

        high_level_graph
    }
}

fn leiden_l1<N: Send + Sync, E: Send + Sync>(
    graph: &Graph<N, E>,
    optimizer: &mut impl ModularityOptimizer,
) -> Graph<Community, ()> {
    let mut community_assignments = graph.initial_community();
    graph.optimize_modularity(&mut community_assignments, optimizer);
    let communities = graph.refine(&community_assignments);
    return compress_l1(graph, communities);
}

fn leiden_ln(
    graph: Graph<Community, ()>,
    optimizer: &mut impl ModularityOptimizer,
) -> Graph<Community, ()> {
    let mut community_assignments = graph.initial_community();
    graph.optimize_modularity(&mut community_assignments, optimizer);
    let communities = graph.refine(&community_assignments);
    if communities.len() == graph._nodes.len() {
        return graph;
    }
    return compress_ln(graph, communities);
}

fn compress_l1<N, E>(
    graph: &Graph<N, E>,
    relabeled_assignments: Vec<HashSet<usize>>,
) -> Graph<Community, ()> {
    let mut node_to_community: HashMap<usize, u32> = HashMap::new();
    let mut new_graph = Graph::new();

    for (i, community) in relabeled_assignments.into_iter().enumerate() {
        for &node in community.iter() {
            node_to_community.insert(node, i as u32);
        }

        let new_community = Community::L1Community(community);

        let node = new_graph.add_node(new_community);
        debug_assert!(node == i);
        let _ = node;
    }

    let node_count = graph.count_nodes();
    for i in 0..node_count {
        let assigni = node_to_community[&i];
        for &(j, edge_id) in &graph._connections[i] {
            if j <= i {
                continue; // process each undirected edge once
            }
            let assignj = node_to_community[&j];
            if assigni == assignj {
                continue;
            }
            let weight = graph._edges[edge_id].weight;
            new_graph.add_edge(assigni as usize, assignj as usize, (), weight);
        }
    }

    new_graph
}

fn compress_ln<E>(
    mut graph: Graph<Community, E>,
    relabeled_assignments: Vec<HashSet<usize>>,
) -> Graph<Community, ()> {
    let mut node_to_community: HashMap<usize, u32> = HashMap::new();
    let mut new_graph = Graph::new();

    for (i, community) in relabeled_assignments.into_iter().enumerate() {
        for &node in community.iter() {
            node_to_community.insert(node, i as u32);
        }

        let new_community = if community.len() == 1 {
            let &c = community.iter().next().unwrap();
            let x = std::mem::replace(&mut graph._nodes[c], Community::LNCommunity(Vec::new()));
            #[cfg(debug_assertions)]
            {
                if let Community::L1Community(sub_communities) = &x {
                    debug_assert!(sub_communities.len() >= 1);
                }
            }
            x
        } else {
            let sub_communities: Vec<Community> = community
                .into_iter()
                .map(|c| {
                    let x =
                // graph._nodes is taken out
                std::mem::replace(&mut graph._nodes[c], Community::LNCommunity(Vec::new()));
                    x
                })
                .collect();
            debug_assert!(sub_communities.len() >= 1);
            Community::LNCommunity(sub_communities)
        };

        let node = new_graph.add_node(new_community);
        debug_assert!(node == i);
        let _ = node;
    }

    let node_count = graph.count_nodes();
    for i in 0..node_count {
        let assigni = node_to_community[&i];
        for &(j, edge_id) in &graph._connections[i] {
            if j <= i {
                continue; // process each undirected edge once
            }
            let assignj = node_to_community[&j];
            if assigni == assignj {
                continue;
            }
            let weight = graph._edges[edge_id].weight;
            new_graph.add_edge(assigni as usize, assignj as usize, (), weight);
        }
    }

    new_graph
}

#[cfg(test)]
mod tests {
    use crate::{Graph, TrivialModularityOptimizer};
    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_example() {
        let edges: &[(&'static str, &'static str, f32)] = &[
            ("Fortran", "C", 0.5),
            ("Fortran", "LISP", 0.3),
            ("Fortran", "MATLAB", 0.6),
            ("C", "C++", 0.9),
            // ("C", "Java", 0.2),
            ("C", "Go", 0.6),
            ("LISP", "ML", 0.5),
            ("LISP", "OCaml", 0.2),
            ("LISP", "Haskell", 0.2),
            ("LISP", "Ruby", 0.5),
            ("LISP", "Julia", 0.6),
            ("ML", "OCaml", 0.8),
            ("ML", "Haskell", 0.5),
            ("OCaml", "Haskell", 0.3),
            ("OCaml", "F#", 0.6),
            ("Haskell", "Julia", 0.2),
            // ("C++", "Java", 0.5),
            ("C++", "Python", 0.32),
            ("C++", "Ruby", 0.2),
            ("C++", "C#", 0.5),
            // ("Java", "Ruby", 0.4),
            // ("Java", "Python", 0.5),
            // ("Java", "C#", 0.6),
            // ("Java", "Go", 0.45),
            // ("Java", "Julia", 0.1),
            ("Python", "F#", 0.2),
            ("Python", "Julia", 0.4),
            ("C#", "F#", 0.3),
        ];

        let mut nodes: HashMap<&'static str, usize> = HashMap::new();
        let mut g = Graph::new();
        for (from, to, weight) in edges.iter() {
            let from_id = *nodes.entry(from).or_insert_with(|| g.add_node(from));
            let to_id = *nodes.entry(to).or_insert_with(|| g.add_node(to));
            g.add_edge(from_id, to_id, (), *weight);
        }

        let mut optimizer = TrivialModularityOptimizer {
            parallel_scale: 128,
            tol: 1e-11,
        };

        let hierarchy = g.leiden(Some(100), &mut optimizer);
        for (i, node) in hierarchy.node_data_slice().iter().enumerate() {
            println!("community {}:", i);
            node.collect_nodes(&|i| {
                let n = g.node_data_slice()[i];
                println!("     {}", n);
            });
        }
    }

    #[test]
    fn test_weighted_modularity() {
        // Two 3-cliques (intra weight 10.0) connected by a single bridge (weight 1.0).
        // Total weight m = 3*10 + 3*10 + 1 = 61.
        // With the correct 2-community assignment the weighted modularity should be:
        //   Q ≈ 0.6503
        // The unweighted-degree bug produced Q ≈ 0.979, so this test distinguishes them.
        let mut g: Graph<usize, ()> = Graph::new();
        let nodes: Vec<usize> = (0..6).map(|i| g.add_node(i)).collect();

        // Clique A: 0-1-2
        g.add_edge(nodes[0], nodes[1], (), 10.0);
        g.add_edge(nodes[0], nodes[2], (), 10.0);
        g.add_edge(nodes[1], nodes[2], (), 10.0);
        // Clique B: 3-4-5
        g.add_edge(nodes[3], nodes[4], (), 10.0);
        g.add_edge(nodes[3], nodes[5], (), 10.0);
        g.add_edge(nodes[4], nodes[5], (), 10.0);
        // Bridge 2-3
        g.add_edge(nodes[2], nodes[3], (), 1.0);

        // Perfect 2-community assignment: {0,1,2} → 0, {3,4,5} → 1
        let mut assignments = g.initial_community();
        for &n in &nodes[0..3] {
            assignments.insert(n, 0u32);
        }
        for &n in &nodes[3..6] {
            assignments.insert(n, 1u32);
        }

        let q = g.compute_modularity(&assignments);
        // Correct weighted Q ≈ 0.6503; incorrect unweighted Q ≈ 0.979
        assert!(
            (q - 0.6503_f32).abs() < 0.01,
            "expected Q ≈ 0.6503, got {}",
            q
        );
    }

    #[test]
    fn test_simplest() {
        let edges = vec![
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
            (4, 5, 1.0),
            (4, 6, 1.0),
            (5, 6, 1.0),
            (7, 8, 1.0),
            (7, 9, 1.0),
            (8, 9, 1.0),
        ];

        let mut nodes: HashMap<usize, usize> = HashMap::new();
        let mut g = Graph::new();
        for (from, to, weight) in edges.into_iter() {
            let from_id = *nodes.entry(from).or_insert_with(|| g.add_node(from));
            let to_id = *nodes.entry(to).or_insert_with(|| g.add_node(to));
            g.add_edge(from_id, to_id, (), weight);
        }

        let mut optimizer = TrivialModularityOptimizer {
            parallel_scale: 128,
            tol: 1e-13,
        };

        let assignments = RefCell::new(g.initial_community());

        let hierarchy = g.leiden(Some(100), &mut optimizer);
        for (i, node) in hierarchy.node_data_slice().iter().enumerate() {
            println!("community {}:", i);
            let comm = i;
            node.collect_nodes(&|i| {
                assignments.borrow_mut().insert(i, comm as u32);
                let n = g.node_data_slice()[i];
                println!("     {}", n);
            });
        }

        assert!(assignments.borrow().values().collect::<HashSet<_>>().len() == 3);

        println!(
            "real modularity: {}",
            g.compute_modularity(&assignments.borrow())
        );
    }
}
