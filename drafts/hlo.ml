type shape = int list
and op = Parameter | Dot of { lhs: hlo; rhs: hlo; lhs_c: int; rhs_c: int }
and hlo = Node of { op: op; shape: shape; prestine: bool }

let make_dot lhs rhs i j =
  let rec out_shape sl sr i j so =
    match sl, sr, i, j with
    | _ :: sl, sr, 0, j -> out_shape sl sr (i - 1) j so
    | sl, _ :: sr, i, 0 -> out_shape sl sr i (j - 1) so
    | d :: sl, sr, i, j -> out_shape sl sr (i - 1) j (d :: so)
    | [], d :: sr, i, j -> out_shape sl sr i (j - 1) (d :: so)
    | [], [], _, _ -> so
  in match lhs, rhs with
  | Node lhs, Node rhs ->
    assert (List.nth lhs.shape i = List.nth rhs.shape j);
    Node {
      op = Dot { lhs = Node lhs; rhs = Node rhs; lhs_c = i; rhs_c = j };
      shape = out_shape lhs.shape rhs.shape i j [];
      prestine = true;
    }

let rec string_of_shape s =
  match s with
  | [] -> ""
  | d :: s -> Printf.sprintf "%d," d ^ string_of_shape s

let rec string_of_hlo (Node root) =
  match root.op with
  | Parameter -> Printf.sprintf "Parameter { shape=(%s) }" (string_of_shape root.shape)
  | Dot { lhs; rhs; lhs_c; rhs_c } ->
    Printf.sprintf
      "Dot { lhs=%s; rhs=%s; %d %d; shape=(%s) }"
      (string_of_hlo lhs)
      (string_of_hlo rhs)
      lhs_c rhs_c
      (string_of_shape root.shape)
