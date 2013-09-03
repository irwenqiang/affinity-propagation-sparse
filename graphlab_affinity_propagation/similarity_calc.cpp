#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <limits>
#include <vector>
#include <algorithm>
#include <graphlab.hpp>
using namespace std;

struct vertex_data {
	std::vector<double> feature;	
	std::vector<double> sims;
	size_t num;

	void save(graphlab::oarchive& oarc) const {
		oarc << num << feature << sims;
	}

	void load(graphlab::iarchive& iarc) {
		iarc >> num >> feature >> sims;
	}
};

double eucliean(const std::vector<double>& a, const std::vector<double>& b) {
	ASSERT_EQ(a.size(), b.size());

	double sum = 0.0;
	for(size_t i = 0; i < a.size(); i++){	
		sum += (a[i] - b[i]) * (a[i] - b[i]);	
	}

	return -sqrt(sum);
}

typedef graphlab::distributed_graph<vertex_data, graphlab::empty> graph_type;

graphlab::atomic<graphlab::vertex_id_type> NEXT_VID;

bool vertex_loader(graph_type& graph, const std::string& fname, const std::string& line) {

	if (line.empty())
		return true;
	namespace qi = boost::spirit::qi;
	namespace ascii = boost::spirit::ascii;
	namespace phoenix = boost::phoenix;

	vertex_data vtx;

	graphlab::vertex_id_type vid;

	const bool success = qi::phrase_parse
		(line.begin(), line.end(),
		 //  Begin grammar
		 (
		  (qi::double_[phoenix::push_back(phoenix::ref(vtx.feature), qi::_1)] % -qi::char_(",") )
		 )
		 ,
		 //  End grammar
		 ascii::space);

	if (!success) return false;

	vtx.num = size_t(vtx.feature.back());

	vtx.feature.pop_back();

	vid = NEXT_VID.inc_ret_last(graph.numprocs());	
	graph.add_vertex(vid, vtx);		
	graph.add_edge(vid, vtx.num, graphlab::empty());

	return true;
}

struct set_union_gather : public graphlab::IS_POD_TYPE {

	std::vector<double> sims;
	std::vector<double> source_feature;
	std::vector<double> target_feature;
	graphlab::vertex_id_type source;
	graphlab::vertex_id_type target;

	set_union_gather() {}
	explicit set_union_gather(const std::vector<double> &s_feature, const std::vector<double> &t_feature, graphlab::vertex_id_type s, graphlab::vertex_id_type t) {
		source_feature = s_feature;
		target_feature = t_feature; 
		source = s;
		target = t;
	}

	set_union_gather& operator+=(const set_union_gather& other) {	
	
		double sim = eucliean(source_feature, other.target_feature);		
		sims.push_back(sim);	
		return *this;
	}	
};

struct vertex_writer {
	std::string save_vertex(graph_type::vertex_type v) {
		std::stringstream strm;
		for(size_t i = 0; i < v.data().sims.size();i++) {
			strm << v.data().sims[i] << "\n";
		}

		strm.flush();

		return strm.str();
	}

	std::string save_edge(graph_type::edge_type e) {return "";}
};


class Similarity_Calc : public graphlab::ivertex_program<graph_type, set_union_gather>, public graphlab::IS_POD_TYPE {
	public:


		edge_dir_type gather_edges(icontext_type& context,
				const vertex_type& vertex) const {
			return graphlab::ALL_EDGES;
		}


		gather_type gather(icontext_type& context, const vertex_type& vertex, edge_type& edge) const {	
			return set_union_gather(vertex.data().feature, edge.target().data().feature, vertex.id(), edge.target().id());
		}

		void apply(icontext_type& context, vertex_type& vertex,
				const gather_type& total) {
			//vertex.sims = total.sims;
			for (int i = 0; i < total.sims.size(); i++)
				vertex.data().sims.push_back(total.sims[i]);
		}

		edge_dir_type scatter_edges(icontext_type& context,
				const vertex_type& vertex) const {
			return graphlab::OUT_EDGES;
		}

		void scatter(icontext_type& context, const vertex_type& vertex,
				edge_type& edge) const {
		}

};

int main(int argc, char** argv) {


	graphlab::command_line_options clopts
		("K-means clustering. The input data file is provided by the "
		 "--data argument which is non-optional. The format of the data file is a "
		 "collection of lines, where each line contains a comma or white-space "
		 "separated lost of numeric values representing a vector. Every line "
		 "must have the same number of values. The required --clusters=N "
		 "argument denotes the number of clusters to generate. To store the output "
		 "see the --output-cluster and --output-data arguments");

	std::string datafile;
	std::string outcluster_file;
	std::string outdata_file;
	clopts.attach_option("data", datafile,
			"Input file. Each line hold a white-space or comma separated numeric vector");	
	clopts.attach_option("output-clusters", outcluster_file,
			"If set, will write a file containing cluster centers "
			"to this filename. This must be on the local filesystem "
			"and must be accessible to the root node.");
	clopts.attach_option("output-data", outdata_file,
			"If set, will output a copy of the input data with an additional "
			"last column denoting the assigned cluster centers. The output "
			"will be written to a sequence of filenames where each file is "
			"prefixed by this value. This may be on HDFS.");


	if(!clopts.parse(argc, argv)) return EXIT_FAILURE;
	if (datafile == "") {
		std::cout << "--data is not optional\n";
		clopts.print_description();
		return EXIT_FAILURE;
	}

	graphlab::mpi_tools::init(argc, argv);
	graphlab::distributed_control dc;


	// load graph
	graph_type graph(dc, clopts);
	NEXT_VID = dc.procid();
	graph.load(datafile, vertex_loader);

	graph.finalize();

	typedef graphlab::omni_engine<Similarity_Calc> engine_type;

	engine_type engine(dc, graph, "sync", clopts);
	engine.signal_all();
	graphlab::timer timer;
	engine.start();

	//写数据点
	if (!outdata_file.empty()) {
		dc.cout() << "Writing Data and its vesting cluster to file " << outdata_file << std::endl;
		graph.save(outdata_file, vertex_writer(), false, true, false, 1);
	}

	graphlab::mpi_tools::finalize();

}	
