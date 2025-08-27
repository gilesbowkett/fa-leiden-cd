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
    _connections: Vec<HashMap<usize, usize>>,
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
        previous - current < self.tol
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
        self._connections.push(HashMap::new());
        id
    }

    pub fn add_edge(&mut self, n1: usize, n2: usize, edge_data: E, weight: f32) -> Option<usize> {
        if n1 == n2 {
            return None;
        }

        let conn = &self._connections[n1];

        if let Some(edge_id) = conn.get(&n2) {
            let edge_id = *edge_id;
            let old = &mut self._edges[edge_id];
            old.weight += weight;
            self._total_weight += weight;
            old.edge_data = edge_data;
            return Some(edge_id);
        }

        let edge_id = self._edges.len();
        let edge_info = EdgeInfo {
            edge_data,
            weight,
            id: edge_id,
        };
        self._edges.push(edge_info);
        self._connections[n1].insert(n2, edge_id);
        self._connections[n2].insert(n1, edge_id);
        self._total_weight += weight;
        Some(edge_id)
    }

    pub fn count_nodes(&self) -> usize {
        self._nodes.len()
    }

    pub fn try_get_edge_between(&self, n1: usize, n2: usize) -> Option<&EdgeInfo<E>> {
        self._connections[n1]
            .get(&n2)
            .map(|edge_id| &self._edges[*edge_id])
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

        for i in 0..node_count {
            let assigni = get_assignment!(i);
            let conn_i = &self._connections[i];
            let ki = conn_i.len() as f32;
            for j in (i + 1)..node_count {
                let assignj = get_assignment!(j);
                if assigni != assignj {
                    continue;
                }

                let kj = self._connections[j].len() as f32;

                match conn_i.get(&j) {
                    Some(edge_ij) => {
                        let edge_ij = *edge_ij;
                        let edge_ij_weight = self._edges[edge_ij].weight;
                        q += edge_ij_weight - (ki * kj) / (m + m);
                    }
                    None => {
                        q += -ki * kj / (m + m);
                    }
                }
            }
        }

        return q / m;
    }

    /// Move a single node to avoid local minimal
    fn _optimize_modularity_handle_pitfall(
        &self,
        assignments: &mut CommunityAssignments,
        current_modularity: f32,
    ) {
        let node_count = self.count_nodes();
        for i in 0..node_count {
            let node = i;
            if let Some(local_move) = self.fast_local_move(node, assignments, current_modularity) {
                assignments.insert(local_move.node, local_move.community);
            }
        }
    }

    fn optimize_modularity(
        &self,
        assignments: &mut CommunityAssignments,
        optimizer: &mut impl ModularityOptimizer,
    ) {
        let mut current_modularity = self.compute_modularity(assignments);
        let node_count = self.count_nodes();
        let parallel_threshold = optimizer.get_parallel_threshold();
        let mut previous_modularity: f32;

        if node_count < parallel_threshold {
            let mut batch_moving: Vec<LocalMove> = Vec::new();

            loop {
                previous_modularity = current_modularity;

                for i in 0..node_count {
                    let node = i;
                    if let Some(local_move) =
                        self.fast_local_move(node, assignments, current_modularity)
                    {
                        batch_moving.push(local_move);
                    }
                }

                if batch_moving.is_empty() {
                    break;
                } else {
                    for local_move in batch_moving.iter() {
                        assignments.insert(local_move.node, local_move.community);
                    }
                    batch_moving.clear();
                }

                current_modularity = self.compute_modularity(assignments);
                if current_modularity == previous_modularity {
                    // but batch_moving is not empty
                    // in this case we randomly choose a node to move to avoid local minimal
                    self._optimize_modularity_handle_pitfall(assignments, current_modularity);
                }
                if optimizer.is_converged(previous_modularity, current_modularity) {
                    break;
                }
            }
        } else {
            let mut batch_moving: boxcar::Vec<LocalMove> = boxcar::Vec::new();

            loop {
                previous_modularity = current_modularity;

                (0..node_count).into_par_iter().for_each(|node| {
                    if let Some(local_move) =
                        self.fast_local_move(node, assignments, current_modularity)
                    {
                        batch_moving.push(local_move);
                    }
                });

                if batch_moving.is_empty() {
                    break;
                } else {
                    for (_, local_move) in batch_moving.iter() {
                        assignments.insert(local_move.node, local_move.community);
                    }

                    batch_moving.clear();
                }

                current_modularity = self.compute_modularity(assignments);

                if current_modularity == previous_modularity {
                    // but batch_moving is not empty
                    // in this case we randomly choose a node to move to avoid local minimal
                    self._optimize_modularity_handle_pitfall(assignments, current_modularity);
                }

                if optimizer.is_converged(previous_modularity, current_modularity) {
                    break;
                }
            }
        }
    }

    fn fast_local_move(
        &self,
        node: usize,
        assignments: &CommunityAssignments,
        current_modularity: f32,
    ) -> Option<LocalMove> {
        let neighbors = &self._connections[node];
        let mut best_assign = assignments[&node];
        let mut changed = false;
        let mut current_modularity = current_modularity;

        for &neighbor in neighbors.keys() {
            let neighbor_assign = assignments[&neighbor];
            if neighbor_assign == best_assign {
                continue;
            }

            let new_modularity = self.compute_modularity_with_local_move(
                assignments,
                LocalMove {
                    node,
                    community: neighbor_assign,
                },
            );

            if new_modularity > current_modularity {
                best_assign = neighbor_assign;
                current_modularity = new_modularity;
                changed = true;
            }
        }

        if changed {
            Some(LocalMove {
                node,
                community: best_assign,
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

                for neighbor in neighbors.keys() {
                    if !community.contains(neighbor) {
                        // the sub-community shall not get connected via this node
                        continue;
                    }

                    queue.push_back(*neighbor);
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

        for j in i + 1..node_count {
            let assignj = node_to_community[&j];

            if assigni == assignj {
                continue;
            }

            if let Some(edge_info) = graph.try_get_edge_between(i, j) {
                new_graph.add_edge(assigni as usize, assignj as usize, (), edge_info.weight);
            }
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

        for j in i + 1..node_count {
            let assignj = node_to_community[&j];

            if assigni == assignj {
                continue;
            }

            if let Some(edge_info) = graph.try_get_edge_between(i, j) {
                new_graph.add_edge(assigni as usize, assignj as usize, (), edge_info.weight);
            }
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
